"""
webapp.py — Dark vintage Casa California. Download endpoint receives real
form values via query params. All em-dashes removed.
Run: python webapp.py  →  http://127.0.0.1:5001
"""

from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
from flask import Flask, Response, jsonify, request

MODEL_NAME = "CaliforniaHousingPredictor"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

app = Flask(__name__)

print(f"[APP] Loading models:/{MODEL_NAME}/Production ...")
try:
    MODEL = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    print(f"[APP] Model loaded: {type(MODEL).__name__}")
except Exception as e:
    print(f"[APP] Load failed: {e}")
    MODEL = None


LAYOUT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title} 🏠︎ Casa California</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400;1,600&family=DM+Sans:wght@300;400;500;600&family=Caveat:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<style>
  :root {{
    --ink:        #1a1512;
    --ink-2:      #221b16;
    --char:       #2a201a;
    --ember:      #c96a3d;
    --ember-2:    #e08356;
    --gold:       #d4a373;
    --gold-2:     #e8c9a0;
    --cream:      #f0e6d2;
    --cream-2:    #d9cbb3;
    --muted:      #8a7b6b;
    --line:       #3d3128;
    --line-2:     #524438;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--ink);
    color: var(--cream);
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    line-height: 1.65;
    min-height: 100vh;
    overflow-x: hidden;
    cursor: none;
  }}
  .cursor-dot, .cursor-ring {{
    position: fixed; top: 0; left: 0;
    pointer-events: none; z-index: 9999; border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: opacity 0.3s;
  }}
  .cursor-dot {{ width: 6px; height: 6px; background: var(--ember); }}
  .cursor-ring {{
    width: 32px; height: 32px;
    border: 1.5px solid var(--gold);
    transition: transform 0.15s ease-out, width 0.2s, height 0.2s, border-color 0.2s;
  }}
  .cursor-ring.hover {{ width: 52px; height: 52px; border-color: var(--ember); }}
  @media (max-width: 768px) {{
    body {{ cursor: auto; }}
    .cursor-dot, .cursor-ring {{ display: none; }}
  }}

  body::before {{
    content: '';
    position: fixed; inset: 0;
    background-image:
      url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240' viewBox='0 0 240 240'><g stroke='%23c96a3d' stroke-width='0.8' fill='none' opacity='0.10'><circle cx='60' cy='60' r='18'/><circle cx='60' cy='60' r='4' fill='%23d4a373'/><path d='M180 30 Q 210 60 180 90'/><path d='M170 180 l 20 -20 l 20 20 l -20 20 z'/><path d='M40 190 q 20 -25 40 0'/><circle cx='120' cy='120' r='2' fill='%23d4a373'/></g></svg>");
    background-size: 240px 240px;
    pointer-events: none; z-index: 0;
  }}
  body::after {{
    content: '';
    position: fixed; inset: 0;
    background:
      radial-gradient(ellipse at 20% 0%, rgba(201,106,61,0.10), transparent 55%),
      radial-gradient(ellipse at 100% 100%, rgba(212,163,115,0.08), transparent 55%),
      radial-gradient(ellipse at 50% 50%, transparent 40%, rgba(26,21,18,0.5));
    pointer-events: none; z-index: 0;
  }}
  body > * {{ position: relative; z-index: 1; }}

  .nav {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 100;
    padding: 1.2rem 2rem;
    backdrop-filter: blur(14px);
    background: rgba(26,21,18,0.55);
    border-bottom: 1px solid var(--line);
  }}
  .nav-inner {{
    max-width: 1200px; margin: 0 auto;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .brand {{
    display: flex; align-items: center; gap: 0.65rem;
    color: var(--cream); text-decoration: none;
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem; font-weight: 700; letter-spacing: 0.02em;
  }}
  .brand svg {{ width: 26px; height: 26px; stroke: var(--ember); fill: none; stroke-width: 1.6; }}
  .nav-links {{ display: flex; gap: 2.2rem; }}
  .nav-links a {{
    color: var(--cream-2); text-decoration: none;
    font-size: 0.85rem; font-weight: 400;
    letter-spacing: 0.12em; text-transform: uppercase;
    position: relative; transition: color 0.25s;
  }}
  .nav-links a::after {{
    content: ''; position: absolute;
    left: 50%; bottom: -6px; width: 0; height: 1px;
    background: var(--ember); transform: translateX(-50%);
    transition: width 0.3s ease;
  }}
  .nav-links a:hover {{ color: var(--ember-2); }}
  .nav-links a:hover::after {{ width: 100%; }}

  main {{ padding-top: 5rem; }}
  .wrap {{ max-width: 1200px; margin: 0 auto; padding: 0 2rem; }}

  .hero {{
    min-height: 100vh;
    display: grid;
    grid-template-columns: 1.1fr 1.1fr;
    gap: 5rem;
    align-items: center;
    padding: 6rem 0 5rem;
  }}
  .hero-eyebrow {{
    font-family: 'Caveat', cursive;
    font-size: 1.45rem; color: var(--gold);
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 0.8rem;
  }}
  .hero-eyebrow::before {{
    content: ''; width: 40px; height: 1px;
    background: var(--gold); display: inline-block;
  }}
  .hero h1 {{
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 5.2rem);
    font-weight: 700; line-height: 1.02;
    letter-spacing: -0.02em;
    color: var(--cream);
    margin-bottom: 2rem;
  }}
  .hero h1 em {{ font-style: italic; font-weight: 400; color: var(--gold); }}
  .hero p.lead {{
    font-size: 1.05rem; color: var(--cream-2);
    max-width: 34rem; line-height: 1.85;
    margin-bottom: 2.5rem; font-weight: 300;
  }}
  .btn {{
    display: inline-flex; align-items: center; gap: 0.7rem;
    padding: 1rem 2rem; background: var(--ember);
    color: var(--ink); text-decoration: none; border: none;
    border-radius: 100px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600; font-size: 0.9rem;
    letter-spacing: 0.05em; cursor: none;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative; overflow: hidden;
  }}
  .btn:hover {{
    background: var(--ember-2); transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(201,106,61,0.35);
  }}
  .btn svg {{ width: 16px; height: 16px; stroke: currentColor; fill: none; stroke-width: 2.2; }}

  .hero-collage {{
    position: relative;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 1.2rem;
  }}
  .collage-frame {{
    position: relative;
    overflow: hidden;
    border-radius: 2px;
    box-shadow:
      0 20px 50px rgba(0,0,0,0.55),
      0 0 0 1px var(--line-2);
  }}
  .collage-frame img {{
    display: block; width: 100%; height: 100%;
    object-fit: cover;
    filter: sepia(0.3) contrast(1.1) brightness(0.85) saturate(0.9);
    transition: transform 0.8s ease, filter 0.6s;
  }}
  .collage-frame:hover img {{
    transform: scale(1.06);
    filter: sepia(0.15) contrast(1.05) brightness(0.95);
  }}
  .collage-a {{
    grid-column: 1 / 2; grid-row: 1 / 3;
    aspect-ratio: 3/5;
  }}
  .collage-b {{
    grid-column: 2 / 3; grid-row: 1 / 2;
    aspect-ratio: 4/3;
  }}
  .collage-c {{
    grid-column: 2 / 3; grid-row: 2 / 3;
    aspect-ratio: 4/3;
  }}
  .collage-caption {{
    position: absolute; bottom: -2rem; right: 0;
    font-family: 'Caveat', cursive;
    font-size: 1.05rem; color: var(--muted);
    transform: rotate(-3deg);
  }}
  .collage-caption::before {{
    content: ''; position: absolute;
    left: -2rem; top: 50%; width: 1.4rem; height: 1px;
    background: var(--muted);
  }}

  .scroll-mark {{
    text-align: center; padding: 3rem 0;
    color: var(--muted);
    font-family: 'Caveat', cursive;
    font-size: 1.2rem; letter-spacing: 0.1em;
  }}
  .scroll-mark span {{ display: inline-block; animation: bob 2.4s ease-in-out infinite; }}
  @keyframes bob {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(8px); }}
  }}

  .sec {{ padding: 7rem 0; position: relative; }}
  .sec-num {{
    font-family: 'Playfair Display', serif;
    font-style: italic; font-weight: 400;
    color: var(--ember); font-size: 1rem;
    letter-spacing: 0.3em; margin-bottom: 1rem;
  }}
  .sec h2 {{
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 4.5vw, 3.8rem);
    font-weight: 600; line-height: 1.08;
    letter-spacing: -0.01em;
    color: var(--cream); margin-bottom: 1rem;
  }}
  .sec h2 em {{ font-style: italic; color: var(--gold); font-weight: 400; }}
  .sec-sub {{
    font-size: 1.05rem; color: var(--cream-2);
    max-width: 42rem; margin-bottom: 4rem; font-weight: 300;
  }}

  .cards {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    border-top: 1.5px dotted var(--line-2);
    border-left: 1.5px dotted var(--line-2);
  }}
  .card {{
    padding: 3rem 2rem;
    border-right: 1.5px dotted var(--line-2);
    border-bottom: 1.5px dotted var(--line-2);
    position: relative; transition: background 0.4s;
  }}
  .card:hover {{ background: rgba(201,106,61,0.04); }}
  .card-icon {{
    width: 48px; height: 48px; margin-bottom: 1.5rem;
    border: 1.5px dotted var(--gold); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    transition: transform 0.4s, border-color 0.4s;
  }}
  .card:hover .card-icon {{ transform: rotate(45deg); border-color: var(--ember); }}
  .card-icon svg {{ width: 20px; height: 20px; stroke: var(--gold); fill: none; stroke-width: 1.4; }}
  .card h3 {{
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 600;
    color: var(--cream); margin-bottom: 0.8rem;
  }}
  .card p {{ color: var(--muted); font-size: 0.94rem; line-height: 1.75; }}

  .carousel {{
    overflow: hidden;
    border-top: 1.5px dotted var(--line-2);
    border-bottom: 1.5px dotted var(--line-2);
    padding: 3rem 0;
    position: relative;
  }}
  .carousel::before, .carousel::after {{
    content: ''; position: absolute;
    top: 0; bottom: 0; width: 120px; z-index: 2; pointer-events: none;
  }}
  .carousel::before {{
    left: 0;
    background: linear-gradient(90deg, var(--ink), transparent);
  }}
  .carousel::after {{
    right: 0;
    background: linear-gradient(-90deg, var(--ink), transparent);
  }}
  .carousel-track {{
    display: flex; gap: 1.5rem;
    animation: scroll-left 45s linear infinite;
    width: max-content;
  }}
  .carousel-track:hover {{ animation-play-state: paused; }}
  @keyframes scroll-left {{
    from {{ transform: translateX(0); }}
    to   {{ transform: translateX(-50%); }}
  }}
  .carousel-item {{
    flex-shrink: 0;
    width: 280px;
    border-radius: 2px;
    overflow: hidden;
    box-shadow: 0 12px 32px rgba(0,0,0,0.5), 0 0 0 1px var(--line-2);
    transition: transform 0.4s;
  }}
  .carousel-item:hover {{ transform: translateY(-4px) scale(1.02); }}
  .carousel-item img {{
    display: block; width: 100%; height: 200px;
    object-fit: cover;
    filter: sepia(0.25) contrast(1.05) brightness(0.85);
    transition: filter 0.4s;
  }}
  .carousel-item:hover img {{ filter: sepia(0.1) contrast(1.05) brightness(0.95); }}
  .carousel-item .cap {{
    padding: 0.8rem 1rem;
    background: var(--char);
    font-family: 'Caveat', cursive;
    font-size: 1rem;
    color: var(--gold);
    border-top: 1px dotted var(--line-2);
  }}

  .quote {{
    padding: 7rem 0; text-align: center;
    border-top: 1.5px dotted var(--line-2);
    border-bottom: 1.5px dotted var(--line-2);
  }}
  .quote blockquote {{
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: clamp(1.6rem, 3.2vw, 2.6rem);
    line-height: 1.35;
    color: var(--cream);
    max-width: 50rem; margin: 0 auto 1.5rem;
    font-weight: 400;
  }}
  .quote blockquote::before {{ content: '"'; color: var(--ember); font-size: 1.2em; }}
  .quote blockquote::after  {{ content: '"'; color: var(--ember); font-size: 1.2em; }}
  .quote cite {{
    font-family: 'Caveat', cursive;
    font-size: 1.25rem; color: var(--gold); font-style: normal;
  }}

  .cta {{
    padding: 7rem 0;
    text-align: center;
    border-top: 1.5px dotted var(--line-2);
    position: relative;
  }}
  .cta .sec-num {{ margin-bottom: 1.5rem; }}
  .cta h2 {{
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 4rem);
    font-weight: 600; line-height: 1.1;
    letter-spacing: -0.02em;
    color: var(--cream);
    max-width: 44rem; margin: 0 auto 2.5rem;
  }}
  .cta h2 em {{ font-style: italic; color: var(--gold); font-weight: 400; }}
  .cta-actions {{
    display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;
  }}
  .btn-ghost {{
    display: inline-flex; align-items: center; gap: 0.6rem;
    padding: 1rem 2rem; background: transparent;
    color: var(--gold);
    border: 1.5px solid var(--gold);
    border-radius: 100px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500; font-size: 0.9rem;
    letter-spacing: 0.05em; text-decoration: none;
    cursor: none; transition: all 0.3s;
  }}
  .btn-ghost:hover {{
    background: var(--gold); color: var(--ink);
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(212,163,115,0.3);
  }}
  .btn-ghost svg {{ width: 16px; height: 16px; stroke: currentColor; fill: none; stroke-width: 2; }}

  .predict-wrap {{
    border: 1.5px dotted var(--line-2);
    padding: 3.5rem;
    background: rgba(34,27,22,0.4);
    position: relative;
  }}
  .form-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 1.6rem;
  }}
  .field label {{
    display: block; font-size: 0.74rem;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--gold); margin-bottom: 0.5rem; font-weight: 500;
  }}
  .field input {{
    width: 100%; padding: 0.85rem 0;
    background: transparent; border: none;
    border-bottom: 1px solid var(--line-2);
    color: var(--cream);
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem; font-weight: 300;
    transition: border-color 0.3s; cursor: none;
  }}
  .field input:focus {{ outline: none; border-bottom-color: var(--ember); }}
  .form-actions {{ margin-top: 3rem; }}
  .result-box {{
    margin-top: 3rem; padding: 3rem 2rem;
    text-align: center;
    border: 1.5px dotted var(--gold);
    border-radius: 4px;
    background: rgba(212,163,115,0.05);
    display: none; position: relative;
  }}
  .result-box.show {{ display: block; }}
  .result-box small {{
    display: block; font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--gold);
    margin-bottom: 0.8rem;
  }}
  .result-price {{
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 700; color: var(--cream);
    letter-spacing: -0.02em; margin-bottom: 0.5rem;
  }}
  .result-price em {{ color: var(--ember); font-style: normal; }}
  .result-note {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }}
  .download-btn {{
    display: inline-flex; align-items: center; gap: 0.6rem;
    padding: 0.85rem 1.6rem; background: transparent;
    color: var(--gold); border: 1.5px solid var(--gold);
    border-radius: 100px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem; letter-spacing: 0.1em;
    text-transform: uppercase; text-decoration: none;
    font-weight: 500; cursor: none; transition: all 0.3s;
  }}
  .download-btn:hover {{
    background: var(--gold); color: var(--ink);
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(212,163,115,0.3);
  }}
  .download-btn svg {{ width: 15px; height: 15px; stroke: currentColor; fill: none; stroke-width: 2; }}
  .result-box.error {{
    border-color: #b85a4a; background: rgba(184,90,74,0.08);
    color: #e89688; font-family: 'DM Sans', sans-serif; font-size: 0.95rem;
  }}

  .about-grid {{
    display: grid; grid-template-columns: 1fr 1.15fr;
    gap: 5rem; align-items: center;
  }}
  .about-image-wrap {{
    position: relative;
    overflow: hidden;
    border-radius: 2px;
    box-shadow: 0 30px 70px rgba(0,0,0,0.55), 0 0 0 1px var(--line-2);
  }}
  .about-image-wrap img {{
    display: block; width: 100%;
    aspect-ratio: 3/4; object-fit: cover;
    filter: sepia(0.4) contrast(1.1) brightness(0.8);
    animation: gentle-float 8s ease-in-out infinite;
    transition: filter 0.6s;
  }}
  .about-image-wrap:hover img {{
    filter: sepia(0.2) contrast(1.05) brightness(0.9);
    animation-play-state: paused;
  }}
  @keyframes gentle-float {{
    0%, 100% {{ transform: scale(1) translate(0, 0); }}
    50%      {{ transform: scale(1.04) translate(1%, -1%); }}
  }}
  .about-body p {{
    color: var(--cream-2); margin-bottom: 1.4rem;
    line-height: 1.9; font-weight: 300; font-size: 1.02rem;
  }}
  .about-body p strong {{ color: var(--cream); font-weight: 500; }}
  .signature {{
    font-family: 'Caveat', cursive;
    font-size: 1.6rem; color: var(--gold);
    margin-top: 2rem; display: block;
  }}

  footer {{
    border-top: 1.5px dotted var(--line-2);
    padding: 4rem 0 2rem; margin-top: 3rem;
  }}
  .foot-grid {{
    display: grid; grid-template-columns: 1.5fr 1fr 1fr;
    gap: 3rem; max-width: 1200px;
    margin: 0 auto 3rem; padding: 0 2rem;
  }}
  .foot-brand {{
    display: flex; align-items: center; gap: 0.6rem;
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; font-weight: 700;
    color: var(--cream); margin-bottom: 1rem;
  }}
  .foot-brand svg {{ width: 22px; height: 22px; stroke: var(--ember); fill: none; stroke-width: 1.6; }}
  .foot-grid p, .foot-grid a {{
    color: var(--muted); font-size: 0.9rem;
    text-decoration: none; line-height: 1.8;
    transition: color 0.2s; font-weight: 300;
  }}
  .foot-grid a:hover {{ color: var(--ember-2); }}
  .foot-col h4 {{
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem; color: var(--gold);
    margin-bottom: 1rem; letter-spacing: 0.14em;
    text-transform: uppercase; font-weight: 500;
  }}
  .foot-col a {{ display: block; margin-bottom: 0.5rem; }}
  .foot-bottom {{
    max-width: 1200px; margin: 0 auto;
    padding: 1.5rem 2rem 0;
    border-top: 1px solid var(--line);
    text-align: center; color: var(--muted);
    letter-spacing: 0.08em;
    font-family: 'Caveat', cursive;
    font-size: 1.05rem;
  }}

  .fade-up {{ opacity: 0; transform: translateY(30px); }}

  @media (max-width: 900px) {{
    .hero {{ grid-template-columns: 1fr; gap: 3rem; padding: 4rem 0 3rem; min-height: auto; }}
    .hero-collage {{ max-width: 500px; margin: 0 auto; }}
    .cards {{ grid-template-columns: 1fr; }}
    .form-grid {{ grid-template-columns: 1fr; }}
    .about-grid {{ grid-template-columns: 1fr; gap: 2.5rem; }}
    .foot-grid {{ grid-template-columns: 1fr; gap: 2rem; }}
    .nav-links {{ gap: 1.2rem; }}
    .nav-links a {{ font-size: 0.75rem; letter-spacing: 0.08em; }}
    .nav-inner {{ flex-wrap: wrap; gap: 0.8rem; }}
    .predict-wrap {{ padding: 2rem 1.5rem; }}
    .wrap {{ padding: 0 1.2rem; }}
    .sec {{ padding: 4rem 0; }}
    .carousel-item {{ width: 220px; }}
  }}
</style>
</head>
<body>
<div class="cursor-dot"></div>
<div class="cursor-ring"></div>

<nav class="nav">
  <div class="nav-inner">
    <a href="/" class="brand">
      <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M9 20v-6h6v6"/>
      </svg>
      Casa California
    </a>
    <div class="nav-links">
      <a href="/">Home</a>
      <a href="/predict">Predict</a>
      <a href="/about">About</a>
    </div>
  </div>
</nav>

<main>{content}</main>

<footer>
  <div class="foot-grid">
    <div>
      <div class="foot-brand">
        <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/>
        </svg>
        Casa California
      </div>
      <p>A quiet helper for the dreamers, the wanderers, and anyone imagining a life under the California sun.</p>
    </div>
    <div class="foot-col">
      <h4>Explore</h4>
      <a href="/">Home</a>
      <a href="/predict">Predict</a>
      <a href="/about">About</a>
    </div>
    <div class="foot-col">
      <h4>Correspond</h4>
      <a href="mailto:hello@casacalifornia.test">hello@casacalifornia.test</a>
      <a href="#">Instagram</a>
      <a href="#">The Journal</a>
    </div>
  </div>
  <div class="foot-bottom">
    Made with care, in the hills above the Pacific. MMXXVI
  </div>
</footer>

<script>
  const dot = document.querySelector('.cursor-dot');
  const ring = document.querySelector('.cursor-ring');
  let mx = 0, my = 0, rx = 0, ry = 0;
  window.addEventListener('mousemove', e => {{
    mx = e.clientX; my = e.clientY;
    dot.style.left = mx + 'px'; dot.style.top = my + 'px';
  }});
  function tick() {{
    rx += (mx - rx) * 0.18;
    ry += (my - ry) * 0.18;
    ring.style.left = rx + 'px'; ring.style.top = ry + 'px';
    requestAnimationFrame(tick);
  }}
  tick();
  document.querySelectorAll('a, button, input').forEach(el => {{
    el.addEventListener('mouseenter', () => ring.classList.add('hover'));
    el.addEventListener('mouseleave', () => ring.classList.remove('hover'));
  }});

  if (window.gsap) {{
    gsap.registerPlugin(ScrollTrigger);
    gsap.from('.hero-eyebrow', {{ y: 20, opacity: 0, duration: 0.8, delay: 0.2 }});
    gsap.from('.hero h1', {{ y: 40, opacity: 0, duration: 1, delay: 0.35 }});
    gsap.from('.hero p.lead', {{ y: 30, opacity: 0, duration: 0.9, delay: 0.55 }});
    gsap.from('.hero .btn', {{ y: 20, opacity: 0, duration: 0.7, delay: 0.75 }});
    gsap.from('.collage-frame', {{
      scale: 0.92, opacity: 0,
      duration: 1, delay: 0.4, stagger: 0.15, ease: 'power3.out'
    }});

    gsap.to('.collage-a', {{
      y: -30,
      scrollTrigger: {{ trigger: '.hero', start: 'top top', end: 'bottom top', scrub: true }}
    }});
    gsap.to('.collage-b', {{
      y: -20,
      scrollTrigger: {{ trigger: '.hero', start: 'top top', end: 'bottom top', scrub: true }}
    }});

    document.querySelectorAll('.sec, .quote, .predict-wrap, .about-grid, .cta').forEach(el => {{
      gsap.from(el, {{
        y: 50, opacity: 0, duration: 1, ease: 'power2.out',
        scrollTrigger: {{ trigger: el, start: 'top 85%' }}
      }});
    }});

    gsap.from('.card', {{
      y: 40, opacity: 0, duration: 0.8, stagger: 0.15, ease: 'power2.out',
      scrollTrigger: {{ trigger: '.cards', start: 'top 80%' }}
    }});
  }}
</script>
{extra_scripts}
</body>
</html>
"""


LANDING = """
<section class="wrap hero">
  <div>
    <div class="hero-eyebrow">a quiet guide to the golden state</div>
    <h1>The dream of a<br><em>California</em> home,<br>gently priced.</h1>
    <p class="lead">
      Describe the neighborhood you are imagining, the streets, the trees,
      the quiet rhythm of the day, and we will sketch a fair estimate of
      what a home there might cost. No pressure, no noise. Only a number to
      begin the conversation.
    </p>
    <a href="/predict" class="btn">
      Begin the sketch
      <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5 12h14M13 6l6 6-6 6"/>
      </svg>
    </a>
  </div>

  <div class="hero-collage">
    <div class="collage-frame collage-a">
      <img src="https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=700&q=80"
           alt="Villa at golden hour" />
    </div>
    <div class="collage-frame collage-b">
      <img src="https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=700&q=80"
           alt="California bungalow" />
    </div>
    <div class="collage-frame collage-c">
      <img src="https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=700&q=80"
           alt="House by the coast" />
    </div>
    <div class="collage-caption">somewhere in the hills, 2026</div>
  </div>
</section>

<div class="scroll-mark"><span>scroll slowly ↓</span></div>

<section class="wrap sec">
  <div class="sec-num">chapter one</div>
  <h2>Three quiet steps<br>between a place and <em>a number.</em></h2>
  <p class="sec-sub">
    A small ritual. Nothing to install, nothing to remember. Just a few
    details about a street and a gentle reply.
  </p>
  <div class="cards">
    <div class="card">
      <div class="card-icon">
        <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 7l9-4 9 4-9 4-9-4z"/><path d="M3 12l9 4 9-4"/><path d="M3 17l9 4 9-4"/>
        </svg>
      </div>
      <h3>Describe the place</h3>
      <p>Median income, the age of the homes, how many rooms people keep, small details that tell a larger story.</p>
    </div>
    <div class="card">
      <div class="card-icon">
        <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>
        </svg>
      </div>
      <h3>Wait a quiet moment</h3>
      <p>Our model reads the neighborhood and returns a figure in a breath. No accounts, no sign-ups, no small print.</p>
    </div>
    <div class="card">
      <div class="card-icon">
        <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 3v18"/><path d="M5 10l7-7 7 7"/>
        </svg>
      </div>
      <h3>Keep it, or let it go</h3>
      <p>Download a small card with your estimate. Take it to a realtor, a lender, a friend, as a beginning, never an ending.</p>
    </div>
  </div>
</section>

<section class="sec" style="padding-bottom:3rem;">
  <div class="wrap">
    <div class="sec-num">from the field</div>
    <h2>Houses we have<br><em>quietly admired.</em></h2>
    <p class="sec-sub">
      A wandering collection. Scroll slowly or let it drift. Each one a
      different answer to the same California question.
    </p>
  </div>
  <div class="carousel">
    <div class="carousel-track">
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=500&q=80" alt=""><div class="cap">a bungalow in Pasadena</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=500&q=80" alt=""><div class="cap">white walls, red tile</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1580587771525-78b9dba3b914?w=500&q=80" alt=""><div class="cap">a cottage by the sea</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?w=500&q=80" alt=""><div class="cap">glass and eucalyptus</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1600585154526-990dced4db0d?w=500&q=80" alt=""><div class="cap">the long driveway</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=500&q=80" alt=""><div class="cap">stucco and sun</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?w=500&q=80" alt=""><div class="cap">a courtyard in Ojai</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1570129477492-45c003edd2be?w=500&q=80" alt=""><div class="cap">the evening porch</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=500&q=80" alt=""><div class="cap">a bungalow in Pasadena</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=500&q=80" alt=""><div class="cap">white walls, red tile</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1580587771525-78b9dba3b914?w=500&q=80" alt=""><div class="cap">a cottage by the sea</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?w=500&q=80" alt=""><div class="cap">glass and eucalyptus</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1600585154526-990dced4db0d?w=500&q=80" alt=""><div class="cap">the long driveway</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=500&q=80" alt=""><div class="cap">stucco and sun</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?w=500&q=80" alt=""><div class="cap">a courtyard in Ojai</div></div>
      <div class="carousel-item"><img src="https://images.unsplash.com/photo-1570129477492-45c003edd2be?w=500&q=80" alt=""><div class="cap">the evening porch</div></div>
    </div>
  </div>
</section>

<section class="quote">
  <div class="wrap">
    <blockquote>
      A house is a story we tell ourselves about how we want to live.
    </blockquote>
    <cite>someone wise, somewhere warm</cite>
  </div>
</section>

<section class="cta wrap">
  <div class="sec-num">shall we begin?</div>
  <h2>Somewhere in California,<br>a door is waiting <em>for you.</em></h2>
  <div class="cta-actions">
    <a href="/predict" class="btn">
      Sketch an estimate
      <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5 12h14M13 6l6 6-6 6"/>
      </svg>
    </a>
    <a href="/about" class="btn-ghost">
      Our story
      <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5 12h14M13 6l6 6-6 6"/>
      </svg>
    </a>
  </div>
</section>
"""


PREDICT = """
<section class="wrap sec">
  <div class="sec-num">chapter two</div>
  <h2>A fair <em>first estimate.</em></h2>
  <p class="sec-sub">
    Fill in what you know about the neighborhood. Leave the rest close to the
    middle and we will meet you halfway.
  </p>

  <div class="predict-wrap">
    <div id="status" style="font-size:0.9rem;color:var(--muted);margin-bottom:2rem;">
      Warming up the model...
    </div>

    <form id="predict-form" hidden>
      <div class="form-grid">
        <div class="field"><label>Median income · × $10k</label>
          <input name="MedInc" type="number" step="0.1" value="5.0" required></div>
        <div class="field"><label>House age · years</label>
          <input name="HouseAge" type="number" step="1" value="25" required></div>
        <div class="field"><label>Average rooms</label>
          <input name="AveRooms" type="number" step="0.1" value="5.0" required></div>
        <div class="field"><label>Average bedrooms</label>
          <input name="AveBedrms" type="number" step="0.1" value="1.0" required></div>
        <div class="field"><label>Population</label>
          <input name="Population" type="number" step="1" value="1000" required></div>
        <div class="field"><label>Average occupancy</label>
          <input name="AveOccup" type="number" step="0.1" value="3.0" required></div>
        <div class="field"><label>Latitude</label>
          <input name="Latitude" type="number" step="0.01" value="34.0" required></div>
        <div class="field"><label>Longitude</label>
          <input name="Longitude" type="number" step="0.01" value="-118.0" required></div>
      </div>

      <div class="form-actions">
        <button type="submit" class="btn" id="submit-btn">
          Sketch the estimate
          <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
            <path d="M5 12h14M13 6l6 6-6 6"/>
          </svg>
        </button>
      </div>
    </form>

    <div id="result" class="result-box"></div>
  </div>
</section>
"""


PREDICT_JS = """
<script>
  const statusEl = document.getElementById("status");
  const formEl   = document.getElementById("predict-form");
  const resultEl = document.getElementById("result");
  const btn      = document.getElementById("submit-btn");

  async function checkModel() {
    try {
      const r = await fetch("/health");
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || "not ready");
      statusEl.style.display = "none";
      formEl.hidden = false;
      if (window.gsap) {
        gsap.from(formEl.querySelectorAll('.field'), {
          y: 20, opacity: 0, duration: 0.6, stagger: 0.06, ease: 'power2.out'
        });
      }
    } catch (e) {
      statusEl.style.color = "var(--ember)";
      statusEl.textContent = "The model is not ready yet, please run the training step first.";
    }
  }

  function buildCardHTML(data, price) {
    const rows = Object.entries(data).map(([k, v]) => `
      <div style="display:flex;justify-content:space-between;padding:0.55rem 0;border-bottom:1px dotted var(--line-2);font-size:0.9rem;">
        <span style="color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;font-size:0.72rem;">${k}</span>
        <span style="color:var(--cream);font-family:'Playfair Display',serif;font-weight:600;">${v}</span>
      </div>`).join('');

    return `
      <div style="max-width:560px;margin:0 auto;text-align:left;">
        <div style="text-align:center;margin-bottom:1.5rem;">
          <div style="font-family:'Caveat',cursive;font-size:1.3rem;color:var(--gold);">Casa California · Estimate Card</div>
          <div style="font-family:'Playfair Display',serif;font-size:3rem;font-weight:700;color:var(--cream);letter-spacing:-0.02em;margin:0.5rem 0;">
            $${price.toLocaleString("en-US", { maximumFractionDigits: 0 })}
          </div>
          <div style="font-size:0.78rem;color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;">
            Sketch · ${new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
          </div>
        </div>
        <div>${rows}</div>
        <div style="text-align:center;margin-top:1.2rem;font-size:0.82rem;color:var(--muted);font-family:'Caveat',cursive;">
          a first sketch, not an appraisal
        </div>
      </div>
    `;
  }

  formEl.addEventListener("submit", async (e) => {
    e.preventDefault();
    btn.disabled = true;
    const old = btn.innerHTML;
    btn.innerHTML = "Sketching...";

    const data = {};
    new FormData(formEl).forEach((v, k) => (data[k] = parseFloat(v)));

    try {
      const r = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const j = await r.json();
      if (!r.ok || j.error) throw new Error(j.error || "prediction failed");

      // Build query string so /download receives the real values
      const qs = new URLSearchParams(data).toString();

      resultEl.className = "result-box show";
      resultEl.innerHTML = buildCardHTML(data, j.price) + `
        <div style="margin-top:2rem;">
          <a href="/download?${qs}" class="download-btn">
            Download estimate card
            <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 3v12M6 11l6 6 6-6M5 21h14"/>
            </svg>
          </a>
        </div>`;

      if (window.gsap) {
        gsap.from(resultEl, { y: 20, opacity: 0, duration: 0.6, ease: 'power2.out' });
      }
    } catch (err) {
      resultEl.className = "result-box show error";
      resultEl.textContent = "Something went quiet: " + err.message;
    } finally {
      btn.disabled = false;
      btn.innerHTML = old;
    }
  });

  checkModel();
</script>
"""


ABOUT = """
<section class="wrap sec">
  <div class="sec-num">chapter three</div>
  <h2>About <em>Casa California.</em></h2>
  <p class="sec-sub">A small tool for a big decision.</p>

  <div class="about-grid">
    <div class="about-image-wrap">
      <img
        src="https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=900&q=80"
        alt="Vintage California bungalow at dusk"
      />
    </div>
    <div class="about-body">
      <p>
        <strong>Casa California</strong> began as a quiet experiment: can a
        simple model read the texture of a neighborhood and return something
        honest? California housing is complicated, the ocean, the hills, the
        long commutes, the trees. We wanted a tool that respected that
        complexity without pretending to solve it.
      </p>
      <p>
        The estimates here are built on public neighborhood data from across
        the state. They are not appraisals, and they are not promises. They
        are a first sketch, the sort of number you might scribble on the
        back of an envelope before a realtor conversation.
      </p>
      <p>
        If a figure looks surprising, it probably is. Use it as a beginning,
        not an ending. Talk to people. Walk the street. Look up at the trees.
        The best estimate of a home is one you have seen with your own eyes.
      </p>
      <span class="signature">The Casa California team</span>
    </div>
  </div>
</section>
"""


def render(title: str, content: str, extra_scripts: str = "") -> Response:
    return Response(LAYOUT.format(title=title, content=content, extra_scripts=extra_scripts),
                    mimetype="text/html")


@app.route("/")
def index():
    return render("Home", LANDING)


@app.route("/predict")
def predict_page():
    return render("Predict", PREDICT, PREDICT_JS)


@app.route("/about")
def about_page():
    return render("About", ABOUT)


@app.route("/health")
def health():
    if MODEL is None:
        return jsonify({"ok": False, "error": "model not loaded"}), 503
    return jsonify({"ok": True})


@app.route("/predict", methods=["POST"])
def predict_api():
    if MODEL is None:
        return jsonify({"error": "model not loaded, run python model.py"}), 503
    try:
        data = request.get_json(force=True)
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"missing fields: {missing}"}), 400
        x = np.array([[float(data[f]) for f in FEATURES]])
        pred = float(MODEL.predict(x)[0])
        return jsonify({"price": round(pred * 100_000, 2), "raw": round(pred, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/download")
def download():
    if MODEL is None:
        return "Model not loaded", 503

    vals = {}
    for f in FEATURES:
        try:
            vals[f] = float(request.args.get(f, 0))
        except (ValueError, TypeError):
            vals[f] = 0.0

    # Reject all-zero inputs (someone typed /download directly)
    if all(v == 0 for v in vals.values()):
        return (
            "No inputs supplied. "
            "Use the predict form to generate a download link.",
            400,
        )

    x = np.array([[vals[f] for f in FEATURES]])
    pred = float(MODEL.predict(x)[0])
    price = round(pred * 100_000, 2)
    today = datetime.now().strftime("%B %d, %Y")

    rows = "".join(
        f'<tr><td style="padding:10px 0;border-bottom:1px dotted #524438;'
        f'text-transform:uppercase;letter-spacing:0.1em;font-size:11px;color:#8a7b6b;">{f}</td>'
        f'<td style="padding:10px 0;border-bottom:1px dotted #524438;text-align:right;'
        f'font-family:Georgia,serif;font-weight:600;color:#f0e6d2;">{vals[f]}</td></tr>'
        for f in FEATURES
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Casa California · Estimate</title></head>
<body style="margin:0;padding:40px;background:#1a1512;font-family:'Helvetica Neue',Arial,sans-serif;color:#f0e6d2;min-height:100vh;">
  <div style="max-width:560px;margin:0 auto;border:1.5px dotted #d4a373;padding:40px 32px;">
    <div style="text-align:center;font-family:Georgia,serif;font-style:italic;font-size:20px;color:#d4a373;margin-bottom:8px;">
      Casa California · Estimate Card
    </div>
    <div style="height:1px;background:#524438;margin:24px 0;"></div>
    <div style="text-align:center;">
      <div style="font-size:11px;letter-spacing:0.2em;text-transform:uppercase;color:#d4a373;margin-bottom:10px;">
        Estimated value
      </div>
      <div style="font-family:Georgia,serif;font-size:56px;font-weight:700;color:#f0e6d2;letter-spacing:-0.02em;">
        ${price:,.0f}
      </div>
      <div style="font-size:11px;letter-spacing:0.14em;text-transform:uppercase;color:#8a7b6b;margin-top:8px;">
        Sketch · {today}
      </div>
    </div>
    <div style="height:1px;background:#524438;margin:24px 0;"></div>
    <table style="width:100%;border-collapse:collapse;">{rows}</table>
    <div style="text-align:center;margin-top:28px;font-family:Georgia,serif;font-style:italic;color:#8a7b6b;font-size:14px;">
      a first sketch, not an appraisal
    </div>
  </div>
</body></html>"""

    filename = f"casa-california-estimate-{datetime.now().strftime('%Y%m%d')}.html"
    return Response(
        html,
        mimetype="text/html",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)