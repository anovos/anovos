Welcome to the Anovos Community! We're excited to bring the immense experience of the Mobilewalla team to the data
science community, and we'd love to have you join us! Here you'll find all the details you need to get started as an
Anovos contributor.

## Getting Started with Anovos

Anovos is an open source project that brings automation to the feature engineering process.

To get Anovos up and running on your local machine, follow
the [Getting Started Guide](https://docs.anovos.ai/gettingstarted)

All repos, sample data, and notebooks you need are located in
the  [Anovos Github Organization](https://github.com/anovos)

## How To Get Involved

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Testing out new features
- Contributing to the docs
- Giving talks about Anovos at Meetups, conferences and webinars

The latest information about how to interact with the project maintainers and broader community is kept
in [COMMUNICATION.md](https://github.com/anovos/communication.md).

## Contribute to Anovos

Pull requests are the best way to propose changes to the codebase. We use GitHub flow and everything happens through
pull requests. We welcome your pull requests; to make it simple, here are the steps to contribute:

- Fork the repo you're updating and create your branch from main.
- If you've added code that should be tested, add tests.
- If you've changed APIs, update the documentation.
- Ensure the test suite passes.
- Issue that pull request!

Any contributions you make will be under the Apache Software License In short, when you submit code changes, your
submissions are understood to be under the same Apache License that covers the project. Feel free to contact the
maintainers if that's a concern.

### Write Thorough Commit Messages

Help reviewers know what you're contributing by writing good commit messages. To standardize how things are done, your
first line of your commit is your _subject_, followed by a blank line, then a message describing what the commit does.
We use the following guidelines suggested [by Chris Beams](https://chris.beams.io/posts/git-commit/):

#### Commit messages

The first line of the commit message is the _subject_, this should be followed by a blank line and then a message
describing the intent and purpose of the commit. These guidelines are based upon
a [post by Chris Beams](https://chris.beams.io/posts/git-commit/).

When you commit, you are accepting our DCO:

> Developer Certificate of Origin
> Version 1.1
>
> Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
> 1 Letterman Drive
> Suite D4700
> San Francisco, CA, 94129
>
> Everyone is permitted to copy and distribute verbatim copies of this
> license document, but changing it is not allowed.
>
> Developer's Certificate of Origin 1.1
>
> By making a contribution to this project, I certify that:
>
> (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
>
> (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
>
> (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
>
> (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.

- When you run `git commit` make sure you sign-off the commit by typing `git commit --signoff` or `git commit -s`
- The commit subject-line should start with an uppercase letter
- The commit subject-line should not exceed 72 characters in length
- The commit subject-line should not end with punctuation (., etc)

Note: please do not use the GitHub suggestions feature, since it will not allow your commits to be signed-off.

When giving a commit body, be sure to:

- Leave a blank line after the subject-line
- Make sure all lines are wrapped to 72 characters

Here's an example that would be accepted:

```
Add luke to the contributors' _index.md file

We need to add luke to the contributors' _index.md file as a contributor.

Signed-off-by: Hans <hans@anovos.ai>
```

Some invalid examples:

```
(feat) Add page about X to documentation
```

> This example does not follow the convention by adding a custom scheme of `(feat)`

```
Update the documentation for page X so including fixing A, B, C and D and F.
```

> This example will be truncated in the GitHub UI and via `git log --oneline`

If you would like to ammend your commit follow this
guide: [Git: Rewriting History](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)

## Report bugs using GitHub's issues

All bugs are tracked using Github issues. If you find something that needs to be addressed, open a new issue; it's easy!

## License

By contributing, you agree that your contributions will be licensed under its Apache License, adhere to the Developer
Certificate of Origin, and adhere to our code of conduct.
