# Publish Bandit Strategy Lab

Prepared for Streamlit Community Cloud on October 10, 2026.

## What you need

- A GitHub account and a repository you administer.
- A Streamlit Community Cloud account connected to GitHub.
- This repository on GitHub, including `app.py` and `requirements.txt`.
- Python **3.13** selected in Cloud's Advanced settings, matching local tests and CI.

Community Cloud runs apps on Linux and installs dependencies from
`requirements.txt`. This simulator needs no API keys, external data, database,
or `packages.txt`. A public GitHub repository is the simplest sharing setup.
A private repository also works with additional GitHub access permissions;
check the deployed app's sharing settings if you use a private repository.

## 1. Create the GitHub repository and push

The local repository is initialized on `main` with an initial commit.

1. Open https://github.com/new while signed in.
2. Name the repository `armed_bandits_sim` and choose its visibility.
3. Leave the README, `.gitignore`, and license initialization options unchecked:
   the local repo already has its own initial history.
4. Create the repository, copy its HTTPS or SSH URL, and run:

```sh
cd "/Users/tom/Programming Repos/armed_bandits_sim"
git remote add origin <YOUR-GITHUB-REPOSITORY-URL>
git push -u origin main
```

Replace the angle-bracket placeholder with the URL copied from GitHub.
Use an existing SSH key or GitHub-supported HTTPS authentication. GitHub account
passwords are not accepted for Git pushes. Do not put credentials in project files.
Alternatively, GitHub Desktop can publish the existing local repository.

No license has been chosen yet. Making a repository public lets people see the
source; select a license if you want to explicitly permit reuse and modification.

## 2. Deploy in Streamlit Community Cloud

1. Sign in at https://share.streamlit.io and connect GitHub if prompted.
2. Switch to the workspace for the GitHub account that owns the repository.
3. Click **Create app**, then **Yup, I have an app**.
4. Enter these settings:

| Setting | Value |
| --- | --- |
| Repository | Your GitHub account / `armed_bandits_sim` |
| Branch | `main` |
| Main file path | `app.py` |
| Python version (Advanced settings) | `3.13` |
| Secrets | Leave empty |
| App URL | Pick an available subdomain, such as `bandit-strategy-lab` |

5. Save Advanced settings and click **Deploy**.
6. Watch the build logs, then open the assigned `https://…streamlit.app` link.
7. Verify app access in a signed-out/private browser window before sharing.

The subdomain is a suggestion; availability is determined during deployment.
Cloud's Python setting must be selected there; `.python-version` helps local
Python tooling but does not replace the Cloud selection.

## 3. Check the hosted app

- Open the app and confirm all six tabs render.
- Record Click / No click in Thompson sampling; inspect the posterior and CSV.
- Switch New / Returning in Contextual Thompson; verify the example chooses B / A.
- Run all three strategies and inspect the comparison charts and download.
- Reset and confirm histories clear. Open a separate browser session and confirm
  it begins with its own fresh histories.

## Maintain and improve it

For routine work, create a branch, change the app, and run tests locally:

```sh
git switch -c improve-lesson
python -m unittest discover -s tests -v
git add app.py README.md
git commit -m "Explain the lesson improvement"
git push -u origin improve-lesson
```

Open a pull request into `main`, check GitHub Actions, and merge when ready.
Community Cloud automatically updates the app when the deployed branch changes.
A failed CI check does not itself stop Cloud deployment: use branches and merge
only after checks pass. Changes to `requirements.txt` trigger dependency installs.
Update the documentation alongside changes to the lessons or runtime.

To undo a bad deployment, use `git revert <COMMIT-SHA>` on `main`, then push the
new revert commit. Cloud picks up that change without rewriting Git history.
For build/runtime failures, open **Manage app** and inspect its logs; reboot the
app after addressing the cause. Changing Cloud's Python version requires deleting
and redeploying the app; retain the URL and deployment settings first.

## Hosting and session limits

Community Cloud is free and has shared compute/memory limits. Inactive apps may
sleep and take time to wake. It suits a teaching/demo app; capacity for large
classes should be checked with actual concurrent viewers.

Each viewer's simulation is kept in server-side Streamlit session state. It is
not a permanent database: a refresh/reconnection, reset, redeploy, or server
restart may clear it. Download CSV before leaving if a run matters. The app caps
histories at 20,000 visitors per strategy and for the contextual model. It does
not collect real ad clicks or real visitor datasets.

`.streamlit/config.toml` disables Streamlit usage-stat gathering; the hosting
provider still operates its own infrastructure. `.gitignore` excludes local
virtual environments, caches, `.env` files, and `.streamlit/secrets.toml`.
If you add API keys later, configure them through Cloud Advanced settings/secrets,
never by committing them.

## Official documentation

- Connect GitHub and required repo permissions: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account
- File layout and dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- Deployment settings: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Updates, logs, and resource limits: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- Python changes: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python
