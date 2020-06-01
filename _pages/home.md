---
title: "Welcome"
layout: splash
permalink: /
header:
  image: /assets/images/home-wallpaper2.jpg
  caption: "Photo by [**NASA**](https://unsplash.com/@nasa)"
welcome_message:
  - excerpt: "I'm a software engineer from Germany with a M.Sc. in Artificial Intelligence, specializing on Deep Learning applications. I started this blog in March 2020 to keep practicing my writing skills and to serve as my personal reference. Writing also allows me to build a deeper understanding of the topics I write about, and ideally my articles may even help or inspire someone else out there."
featured_posts:
  - image_path: assets/images/locality-sensitive-hashing.png
    title: "Locality-Sensitive Hashing for Image Deduplication"
    excerpt: "An approximate algorithm for detecting near-duplicate images."
    url: "/posts/locality-sensitive-hashing"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: assets/images/data-science-curriculum.jpg
    title: "My Data Science Curriculum"
    excerpt: "A collection of online courses and resources on Data Science and Deep Learning."
    url: "/posts/data-science-curriculum"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: assets/images/optimizers-teaser.png
    title: "Neural Network Optimization Algorithms Explained with Code"
    excerpt: "An overview of common optimizers for training neural networks with code."
    url: "/posts/neural-net-optimizers"
    btn_label: "Read More"
    btn_class: "btn--primary"
featured_projects:
  - image_path: assets/images/sports-camera.jpg
    title: "Awesome Sports Camera Calibration"
    excerpt: "A collection of resources on automatic self-calibration of cameras in sports applications."
    url: "/projects/sports-camera-calibration"
    btn_label: "Read More"
    btn_class: "btn--primary"
---

<center><h1>Welcome</h1></center>
{% include feature_row id="welcome_message" type="center" %}

<center><h1>Featured Posts</h1></center>
{% include feature_row id="featured_posts" %}

<center><h1>Featured Projects</h1></center>
{% include feature_row id="featured_projects" %}
