---
title: "Star Players"
description: "A Spotify-esque music player with a 'StarPlay' feature"
date: "Feb 6 2025"
repoURL: "https://github.com/confinlay/star-players"
---

This is a simple project I put together to learn a bit more about frontend development.
It's a Spotify-esque music player with a new feature - _Star Players_.

The idea for the feature came from an observation I had - when organising my music in 
playlists, I would often create two copies of the same playlist. One was all of the songs
which matched the vibe I was going for, and the other was subset of that playlist, 
containing only the songs I loved and wanted to play on repeat. If I was leaving music on in
the background, I would play the longer playlist. Other times, if I had my headphones on and was ready to 
really listen to the music, I would play the playlist with only my faves. 

This gave me an idea for a feature - what if you could just "heart" certain songs in
your playlists, and have a separate shuffle mode for your liked songs? Most music players
already have a "hearting" feature, so it would have to be a star instead. Hence _Star Players_!

I built a simple demo of the feature, with astro, react, and tailwind. It consists of a 
playlist of songs, along with the usual media player features. The star icon next to each
song title allows you to add them to your _Star Players_, and separate shuffle mode in 
the media player allows you to initiate _StarPlay_.

You can check out the demo [here](https://conorfinlay.me/star-players).