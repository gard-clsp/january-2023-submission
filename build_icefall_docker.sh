#!/usr/bin/env bash
# Build a Docker image for K2 snwofall with armory 0.15.3
framework=k2-icefall
version=0.15.3
docker build --force-rm --file docker/JHUM_Armory_K2_Icefall_Dockerfile --build-arg armory_version=${version} --target armory-k2-icefall -t yshao1995/armory-${framework}:${version} .
