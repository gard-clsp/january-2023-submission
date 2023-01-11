# Overview

Armory Sender is a tool for submitting scenario configurations, dockerfiles,
and saved model weights for evaluation.

# Installation

You have received three files from TwoSix containing:

  1. This document: `armory-upload.md`
  2. A `GARD Evaluation 6 Submission Documentation.docx` Word file, and
  3. `armory_upload-6.1.0-py3-none-any.whl` a Python wheel archive containing the armory-upload program

To install in a Python 3.7+ environment type:

    $ pip install armory_upload-6.1.0-py3-none-any.whl

It is usually easiest to install in a python virtual environment, but your local
customs will dictate. This will install the program `armory-upload` and a couple
small python libraries on which it depends. You can confirm the installation
by running

    $ armory-upload --help

Unlike in prior submission periods, you no longer need a `sender.toml` file,
so we didn't include one.  You will need to know your four-letter performer
code (for example, `GARD`).


# Submission Checklist

  1. Performers must submit a defense summary in `GARD Evaluation 6 Submission
     Documentation.docx` to `armory@twosixlabs.com`. Filling out this document will help
     you organize your materials and communicate your submission to us.
  2. If you plan on using a private GitHub repo as an external dependency in your
     evaluation you must add the GitHub user account
     [`armory-twosix`](https://github.com/armory-twosix) as a read access collaborator
     to your repo. Repository access controls are at
     https://github.com/your-repo-name/settings/access. We *strongly* recommend
     that you [make your repository private][gh-private].
  3. Submit custom Docker images (if needed by your experiment)
  4. Submit custom Dockerfiles for your defense with armory-upload (if you uploaded an image)
  5. Submit custom model weights (if used by your experiment)
  6. Submit your config (aka "experiment") file or files

For steps 3–6, the steps are shown below.

  [gh-private]: https://docs.github.com/en/latest/github/administering-a-repository/setting-repository-visibility#making-a-repository-private

# Usage

The examples below assume that the performer code is `GARD`; yours is different.

⚠️: Submitted items are stored under the exact filenames given to `armory-upload`. If
you make multiple submissions with the same filename the earlier submissions will be
overwritten.

⚠️: If you are submitting an experiment (aka "config") that relies on saved model
weights or docker images that you will be submitting, you should only submit the
experiment *after* submitting all other items. If you submit an experiment before
submitting its dependencies the scenario may fail automated validation due to the
absence of those dependencies.

The armory-upload progam has a simple interface, running `armory-uploader --help` shows

    armory-upload version 6.1.0
    usage: armory-upload [-h] CODE submitter file_type file

    Upload a GARD submission file to S3

    positional arguments:
    CODE        performer code
    submitter   submitter email address
    file_type   file type, one of: config, dockerfile, docker-image, model-weights
    file        file to upload

In the examples below, the performer code is `GARD` and the email is `msw@example.com`;
please use your own CODE and email address. We've also been calling the JSON config file
that describes an Armory run an "experiment" because "config" is a tad generic; the two
terms are interchangeable.

In prior evaluation periods we asked that performers put their CODE on the front of each
submitted file. That is no longer required, but it is also accepted.

## Submitting a custom Docker image

If your experiment uses a custom docker image, we need to have that to evaluate your
submission.  In past evaluations, we had accepted the `Dockerfile` itself but have found
them difficult to re-build from that description. We now ask that you send the image
itself.

To prepare you image for upload, get its full name with `docker images`:

    $ docker images
    REPOSITORY             TAG   IMAGE ID       CREATED        SIZE
    example/eval5          1.0   9b319913bbe3   2 weeks ago    13GB

and save the image to a tarfile using `docker save`

    $ docker save -o example-eval5-1.0.tar example/eval5:1.0

This creates a tarfile in the current directory. Docker save puts the image name
and tag into the save file, so `example-eval5-1.0` is just suggestive of the
name it will unpack to; that is, it is a human friendly name.  The size of
the output tarfile will be about that which `docker images` lists, 13GB or more
typically.

To upload the saved image:

    armory-upload JHUM tthebau1@jh.edu docker-image example-eval5-1.0.tar

In eval5 and onward we've improved the upload performance dramatically by
removing an intermediate server. We've measured speeds of up to 2GB / minute which
uploads a docker image in ~6 minutes. Contrasted with some performers seeing
over 12 hours to upload an image, this is an improvement.


## Submitting a Dockerfile

If you submit a docker-image, we ask that you also submit the Dockerfile
that was used to create it. When debugging and integrating submissions, this
can be quite helpful. Assuming your Dockerfile has the default name, run:

    $ armory-upload GARD msw@example.com my-build-directory/Dockerfile

If you have multiple Dockerfiles sent by the same submitter email, you
should name them distinctly since `my-carla-build-directory/Dockerfile`
would overwrite the first when uploaded.  Something like

    $ cp my-carla-build-directory/Dockerfile Dockerfile-carla
    $ armory-upload … Dockerfile-carla

would avoid collisions. This applies to all files uploaded under the same
performer email; they need to be unique in their names.  This is usually
true for past submissions, but in the case of the constant name `Dockerfile`
it might not be.

## Submitting saved model weights

To upload model weights:

    $ armory-upload GARD msw@example.com model-weights msw_carla_weights.pth

As noted in the docker-image upload, you can expect vastly better performance
on huge file uploads, perhaps ~1GB per minute is common depending on your
local connection. These now go directly to S3 and are not funneled thorough
our server any longer.

## Submitting an experiment ("config") file for evaluation

Once the supporting docker-images and model-weights have been uploaded if needed,
submit your JSON experiment config with

    $ armory-upload GARD msw@example.com config my_mnist_defended_eps-1.0.json

Shortly after that upload completes, the Armory submission processor will send
a brief mail to the submitter email address you provided acknowledging our receipt
of your uploads.

As the integration proceeds, you will receive some automated notifications about
successes or failures.
