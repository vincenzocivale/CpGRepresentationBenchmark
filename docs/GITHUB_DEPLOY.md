# GitHub deployment

The GitHub connection used while building this project can read `vincenzocivale/MehylPredictor` and `vincenzocivale/MethylomeReconstruction`, but GitHub rejected branch/content writes with:

```text
403 Resource not accessible by integration
```

Therefore this session could not create/update the remote repository. The delivered project is a complete local Git repository and a Git bundle.

## From the source archive

Create an empty GitHub repository named `CpGRepresentationBenchmark`, then:

```bash
cd CpGRepresentationBenchmark
git remote add origin git@github.com:vincenzocivale/CpGRepresentationBenchmark.git
git push -u origin main
```

## From the `.bundle`

```bash
git clone CpGRepresentationBenchmark.bundle CpGRepresentationBenchmark
cd CpGRepresentationBenchmark
git remote add origin git@github.com:vincenzocivale/CpGRepresentationBenchmark.git
git push -u origin main
```

If the GitHub App is later granted contents-write permission, the same repository can be pushed/updated directly through the connector.
