use core::f32;
use std::io::{stdin, stdout, Write};
use std::process::exit;
use std::thread;
use std::time::Duration;

use cpal::SupportedBufferSize;
use cpal::{
    traits::{DeviceTrait, HostTrait},
    Device,
};
use cpal::{SampleRate, StreamConfig};
use crossterm::cursor::{self, MoveTo, Show};
use crossterm::event::{self, KeyCode, KeyModifiers};
use crossterm::style::{PrintStyledContent, Stylize};
use crossterm::terminal;
use crossterm::terminal::ClearType::FromCursorDown;
use crossterm::{
    cursor::{Hide, RestorePosition, SavePosition},
    execute,
    style::Print,
    terminal::{disable_raw_mode, enable_raw_mode, Clear},
    ExecutableCommand,
};
use ringbuf::traits::{Consumer, Producer, Split};

fn choose_device(ctx: &mut TuiContext, device_options: &Vec<Device>) -> std::io::Result<Device> {
    let prompt = "please select an audio input device".to_owned();
    let options: Vec<_> = device_options
        .iter()
        .map(|d| d.name().unwrap_or("No name".to_owned()))
        .collect();

    let choice = loop {
        if let Some(idx) = ctx.list_picker(&prompt, &options, Duration::from_millis(30))? {
            break idx;
        };
    };

    let chosen_device = device_options[choice].clone();
    Ok(chosen_device)
}

fn report(s: &str) {
    execute!(stdout(), PrintStyledContent(s.blue())).unwrap();
}

/* To repeatedly print to a section of the console, we need to know how many rows we need up front. */

struct TuiContext {
    user_input: Vec<char>,
    user_err_msg: String,
}

impl TuiContext {
    /// Display a list
    /// Allow users to pick from indices
    /// Return the index of their selection
    /// Assume the terminal is tall enough for now
    fn list_picker(
        &mut self,
        prompt: &String,
        options: &Vec<String>,
        poll_duration: Duration,
    ) -> std::io::Result<Option<usize>> {
        let top_row = {
            let (_, h) = terminal::size()?;
            let (_, r) = cursor::position()?;
            let free_rows = h - r;
            let required_rows = options.len() + 2;
            let lines_to_clear = required_rows.saturating_sub(free_rows as usize);

            // clear space
            stdout().execute(Print("\n".repeat(lines_to_clear)))?;

            r.saturating_sub(lines_to_clear as u16)
        };

        stdout().execute(MoveTo(0, top_row))?;
        stdout().execute(Hide)?;
        stdout().execute(Clear(FromCursorDown))?;

        // print prompt
        execute!(stdout(), Print(prompt), Print("\n"))?;

        // print options
        for (i, s) in options.iter().enumerate() {
            stdout().execute(Print(format!("- ({}) {}\n", i, s)))?;
        }

        stdout().flush()?;
        execute!(
            stdout(),
            Print(self.user_err_msg.clone()),
            Print(self.user_input.iter().collect::<String>())
        )?;
        stdout().flush()?;

        enable_raw_mode()?;
        // query input
        if event::poll(poll_duration).unwrap() {
            if let event::Event::Key(event) = event::read().unwrap() {
                log::info!("{:?}", event);
                if event.code == KeyCode::Char('c') && event.modifiers == KeyModifiers::CONTROL {
                    disable_raw_mode()?;
                    exit(0);
                }
                match event.code {
                    KeyCode::Backspace => {
                        self.user_input.pop();
                    }
                    KeyCode::Enter => {
                        let as_str: String = self.user_input.drain(..).collect();
                        if let Ok(idx) = as_str.parse::<usize>() {
                            if idx < options.len() {
                                disable_raw_mode()?;
                                cleanup();
                                self.user_err_msg.clear();
                                return Ok(Some(idx));
                            } else {
                                self.user_err_msg =
                                    format!("'{}' out of range; please try again: ", idx);
                            }
                        } else {
                            self.user_err_msg =
                                format!("'{}' is not a valid index; please try again: ", as_str)
                        }
                    }
                    KeyCode::Char(c) => self.user_input.push(c),
                    _ => {}
                };
            };
        }
        disable_raw_mode()?;

        // print input line
        stdout().execute(MoveTo(0, top_row))?;
        Ok(None)
    }
}

fn choose_channel(ctx: &mut TuiContext, chosen_device: &Device) -> std::io::Result<usize> {
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

    let prompt = "please select a channel".to_owned();
    let mut options = Vec::<String>::with_capacity(num_channels);
    options.resize(num_channels, String::new());
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

        for chan in 0..num_channels {
            const MIN_DB: f32 = -71.;
            const MAX_DB: f32 = 0.;
            let proportion_cur = (db_by_chan[chan].clamp(MIN_DB, MAX_DB) - MIN_DB) / (-MIN_DB);
            let proportion_max = (max_db_by_chan[chan].clamp(MIN_DB, MAX_DB) - MIN_DB) / (-MIN_DB);

            const BAR_WIDTH: usize = 12;
            let pos_cur = (proportion_cur * BAR_WIDTH as f32).floor() as usize;
            let pos_max = (proportion_max * BAR_WIDTH as f32).floor() as usize;
            let mut meter = [' '; 12];
            for i in 0..pos_cur {
                meter[i] = '-';
            }
            meter[pos_cur] = 'o';
            meter[pos_max] = 'm';

            let meter_string: String = meter.iter().collect();

            options[chan] = format!(
                "channel #{} — |{meter_string} | {:.1} (max: {:.1})",
                chan + 1,
                db_by_chan[chan],
                max_db_by_chan[chan]
            );
        }
        if let Some(idx) = ctx.list_picker(&prompt, &options, Duration::from_millis(30))? {
            break idx;
        }
    };

    Ok(choice)
}

/// Be a good citizen and undo our terminal modifications.
fn cleanup() {
    stdout().execute(Show).unwrap();
}

fn main() -> std::io::Result<()> {
    stdout().execute(Clear(crossterm::terminal::ClearType::FromCursorDown))?;
    let mut ctx = TuiContext {
        user_input: vec![],
        user_err_msg: String::new(),
    };

    report("-: tina tuner v1.0 :-\n");
    stdout().execute(Hide)?.execute(SavePosition)?;

    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .expect("there was a problem during audio device detection");
    let device_options: Vec<_> = devices
        .into_iter()
        .filter(|d| d.default_input_config().is_ok())
        .collect();

    let chosen_device: Device = choose_device(&mut ctx, &device_options)?;

    execute!(stdout(), RestorePosition, Clear(FromCursorDown))?;
    report("selected device: ");
    execute!(stdout(), Print(chosen_device.name().unwrap() + "\n"))?;

    stdout().execute(SavePosition)?;

    let chosen_channel = choose_channel(&mut ctx, &chosen_device)?;

    execute!(stdout(), RestorePosition, Clear(FromCursorDown))?;
    report("selected channel: ");
    stdout().execute(Print(format!("{}", chosen_channel + 1) + "\n"))?;

    cleanup();
    Ok(())
}
