use core::f32;
use std::io::{stdin, stdout, Write};
use std::str::FromStr;
use std::thread;
use std::time::Duration;

use cpal::SupportedBufferSize;
use cpal::{
    traits::{DeviceTrait, HostTrait},
    Device,
};
use cpal::{SampleRate, StreamConfig};
use crossterm::cursor::{self, MoveTo};
use crossterm::event::{self, KeyCode, KeyEvent};
use crossterm::style::{PrintStyledContent, Stylize};
use crossterm::terminal;
use crossterm::{
    cursor::{Hide, RestorePosition, SavePosition},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{disable_raw_mode, enable_raw_mode, Clear},
    ExecutableCommand,
};
use ringbuf::traits::{Consumer, Producer, Split};

fn choose_device(device_options: &Vec<Device>) -> std::io::Result<&Device> {
    execute!(stdout(), Print("please select an audio input device\n"))?;
    for (i, d) in device_options.iter().enumerate() {
        execute!(
            stdout(),
            Print(format!(
                "- ({i}) {}\n",
                d.name().unwrap_or("No name".to_owned())
            ))
        )?;
    }

    disable_raw_mode()?;
    let choice = loop {
        let mut input = String::new();
        stdin().read_line(&mut input).expect("Failed to read line");
        let input = input.trim();
        match i32::from_str(input) {
            Ok(num) if num >= 0 && num < device_options.len() as i32 => break num as usize,
            _ => {
                enable_raw_mode()?;
                stdout().execute(Print("please enter a valid number"))?;
                disable_raw_mode()?;
            }
        };
    };
    disable_raw_mode()?;

    let chosen_device = &device_options[choice];
    Ok(chosen_device)
}

fn report(s: &str) {
    execute!(stdout(), PrintStyledContent(s.blue())).unwrap();
}

fn main() -> std::io::Result<()> {
    stdout().execute(Clear(crossterm::terminal::ClearType::FromCursorDown))?;
    stdout().execute(Hide)?.execute(SavePosition)?;

    report("-: tina tuner v1.0 :-\n");

    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .expect("there was a problem during audio device detection");
    let device_options: Vec<_> = devices
        .into_iter()
        .filter(|d| d.default_input_config().is_ok())
        .collect();

    let chosen_device: &Device = choose_device(&device_options)?;

    execute!(
        stdout(),
        RestorePosition,
        Clear(crossterm::terminal::ClearType::FromCursorDown),
        PrintStyledContent("selected device: ".grey()),
        Print(chosen_device.name().unwrap() + "\n")
    )?;

    let chosen_channel = choose_channel(chosen_device)?;
    report("selected channel: ");
    stdout().execute(Print(format!("{}", chosen_channel + 1)))?;

    thread::sleep(Duration::from_millis(30));

    Ok(())
}

fn choose_channel(chosen_device: &Device) -> std::io::Result<usize> {
    const SAMPLE_RATE: SampleRate = SampleRate(48_000);
    let config_range_48k = chosen_device
        .supported_input_configs()
        .unwrap()
        .find(|c| c.min_sample_rate() == SAMPLE_RATE)
        .expect("couldn't find a supported config with sample rate 48000");

    let max_channels = config_range_48k.channels();
    let supported_buffer_size = config_range_48k.buffer_size();
    let buffer_size = match supported_buffer_size {
        SupportedBufferSize::Range { min, max } => {
            if *min <= 512 && *max >= 512 {
                512
            } else {
                *max
            }
        }
        SupportedBufferSize::Unknown => 512,
    };

    let stream_config = StreamConfig {
        channels: max_channels,
        sample_rate: SAMPLE_RATE,
        buffer_size: cpal::BufferSize::Fixed(buffer_size),
    };

    let sample_buffer_size = buffer_size as usize * max_channels as usize;

    // create buffer to copy audio data into
    let rb = ringbuf::HeapRb::<f32>::new(sample_buffer_size);
    let (mut tx, mut rx) = rb.split();

    // build input stream with max channels
    let _input_stream = chosen_device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                tx.push_slice(data);
            },
            move |err| eprintln!("{:?}", err),
            None,
        )
        .expect("failed to build input stream during channel selection");

    let num_channels = max_channels as usize;
    let num_samples = buffer_size as usize;

    let mut sample_buffer = Vec::<f32>::with_capacity(sample_buffer_size);
    sample_buffer.resize(sample_buffer_size, 0.);
    let mut db_by_chan = Vec::<f32>::with_capacity(num_channels);
    db_by_chan.resize(num_channels, 0.);

    let mut max_db_by_chan = Vec::<f32>::with_capacity(num_channels);
    max_db_by_chan.resize(num_channels, f32::NEG_INFINITY);

    let (_, mut top_row) = cursor::position()?;
    let (_, height) = terminal::size()?;

    if height - top_row <= 2 {
        top_row -= 2;
    }

    // make room
    for _ in 0..=num_channels {
        execute!(stdout(), Print("\n"))?;
    }
    stdout().execute(MoveTo(0, top_row))?;
    stdout().execute(Clear(crossterm::terminal::ClearType::FromCursorDown))?;
    stdout().execute(PrintStyledContent("please select a channel".grey()))?;

    let choice = loop {
        rx.pop_slice(&mut sample_buffer[0..sample_buffer_size]);
        // create db meters for each channel
        for chan_idx in 0..num_channels {
            db_by_chan[chan_idx] = 0.;

            for samp_idx in 0..num_samples {
                let i = samp_idx * num_channels + chan_idx;
                db_by_chan[chan_idx] += sample_buffer[i].powi(2);
            }

            db_by_chan[chan_idx] /= num_samples as f32;
            db_by_chan[chan_idx] = db_by_chan[chan_idx].sqrt();

            if db_by_chan[chan_idx] == 0. {
                db_by_chan[chan_idx] = f32::NEG_INFINITY;
            } else {
                db_by_chan[chan_idx] = 20. * db_by_chan[chan_idx].log10();
            }

            max_db_by_chan[chan_idx] = max_db_by_chan[chan_idx].max(db_by_chan[chan_idx]);
        }

        stdout().execute(MoveTo(0, top_row + 1))?;

        for chan in 0..num_channels {
            const MIN_DB: f32 = -71.;
            const MAX_DB: f32 = 0.;
            let proportion_cur = (db_by_chan[chan].clamp(MIN_DB, MAX_DB) - MIN_DB) / (-MIN_DB);
            let proportion_max = (max_db_by_chan[chan].clamp(MIN_DB, MAX_DB) - MIN_DB) / (-MIN_DB);

            const BAR_WIDTH: usize = 12;
            let pos_cur = (proportion_cur * BAR_WIDTH as f32).floor() as usize;
            let pos_max = (proportion_max * BAR_WIDTH as f32).floor() as usize;
            let mut meter = [' '; 12];
            for i in 0..=pos_cur {
                meter[i] = '|';
            }
            meter[pos_max] = 'M';

            let meter_string: String = meter.iter().collect();

            execute!(
                stdout(),
                Print(format!(
                    "- ({:01x}) channel #{} — |{meter_string} | {:.1} (max: {:.1})\n",
                    chan,
                    chan + 1,
                    db_by_chan[chan],
                    max_db_by_chan[chan]
                )),
            )?;
        }

        stdout().flush()?;

        let (_, input_row) = cursor::position()?;

        if event::poll(Duration::from_millis(30)).unwrap() {
            if let event::Event::Key(KeyEvent {
                code: KeyCode::Char(c),
                ..
            }) = event::read().unwrap()
            {
                let is_digit = ('0'..='9').contains(&c);
                let is_letter = ('a'..='f').contains(&c);
                if is_digit || is_letter {
                    let as_idx = if is_digit {
                        c as usize - '0' as usize
                    } else {
                        10 + c as usize - 'a' as usize
                    };

                    if as_idx < num_channels {
                        break as_idx;
                    } else {
                        stdout().execute(MoveTo(0, input_row))?;
                        println!(
                            "channel {} out of range. please select a channel between {} and {}",
                            c,
                            0,
                            num_channels - 1,
                        );
                    }
                }
            }
        }
    };

    Ok(choice)
}
