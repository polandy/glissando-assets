# Glissando demo media

The media of the demo slideshow that the public install (https://polandy.github.io/glissando/)
offers on an empty library. `glissando-demo.glissando` is the slideshow itself, exported from
Glissando; the app downloads it on demand.
It is format version 22 and carries each picture's found focus, so opening it needs no focus
search; an app older than polandy/glissando#331 refuses it as newer.

## Rights

- `glissando-demo.glissando` — the photos and video clips in it: © polandy, all rights reserved,
  used as the Glissando demo only. They are **not** covered by any open-source licence of this or
  any other repository. Its music ("Hold the Frame") is AI-generated, designed by polandy. The
  file carries no location data, camera serial numbers or Immich asset ids.
- `music/` — the track `glissando-demo.mp3` and its generator `render_demo_track.py`
  (`python3 render_demo_track.py out.wav`, needs NumPy). Every sound is synthesised by the
  script; no samples, no third-party material. Licensed under the MIT licence of
  https://github.com/polandy/glissando.
