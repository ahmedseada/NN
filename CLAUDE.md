# Working on this branch (Qasd)

- Qasd uses the Idrak library only through its NuGet packages (nuget.org) and the `idrak-tune` tool
  (`.config/dotnet-tools.json`). No project references to the library, no copied library files: a library fix goes to
  https://github.com/ahmedseada/Idrak and arrives here as a new package version.
- Commands for the user (Windows, PowerShell) start with `git pull origin Qasd`.
- This container has no GPU: say which parts were only checked on the CPU.
