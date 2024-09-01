# tina tuner

A tiny chromatic tuner for the command line.

## usage

### select an audio source

```
~ tina
-: tina tuner v1.0 :-

please select an audio source (2 sources available)
(1) Built-in Audio
(2) Babyface

>>> 2
```

### select a channel

```
please select a channel (4 channels available)
(1) |           | (-inf db) (-inf db)
(2) |-----o   | | (-4   db) (-12  db)
(3) |           | (-inf db) (-inf db)
(4) |           | (-inf db) (-inf db)

>>> 2
```

### tune

```
pressed ctrl+D to exit

-50c              D              +50c
  |---o----------(-)--------------|
```
