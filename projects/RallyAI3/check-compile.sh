#!/usr/bin/env bash
# Semantic compile check for the project's C#, without opening Unity.
#
# WHY THIS SCRIPT LOOKS LIKE THIS
#
# The tempting version — run a compiler and grep its output for lines starting with
# "Assets" — reports "clean" on code that does not compile, and it did exactly that
# on this project for a while.
#
# Two traps, both real, both hit here:
#
#  1. Wrong compiler. mcs is a Mono compiler; Unity 6 compiles against .NET Standard
#     2.1 reference assemblies. mcs cannot resolve them and dies with CS1070 ("type
#     forwarded to an assembly that is not referenced"). CS1070 is FATAL — the
#     compiler stops before full semantic analysis, so every genuine error after that
#     point is never reached and never printed.
#
#  2. Grepping instead of trusting the exit code. Filtering CS1070 away as "noise"
#     hides the fact that the check aborted. A missing `using` sailed straight
#     through and only surfaced when Unity refused to build the scene.
#
# So: use the Roslyn and the reference assemblies that Unity itself uses, and report
# the compiler's exit code. Never grep for success.
#
# Usage:  ./check-compile.sh
# Exit:   0 = compiles, 1 = does not
set -uo pipefail

U="/Applications/Unity/Hub/Editor/6000.4.11f1/Unity.app/Contents"
DOTNET="$U/Resources/Scripting/NetCoreRuntime/dotnet"
CSC="$U/Resources/Scripting/DotNetSdkRoslyn/csc.dll"
NETSTD="$U/Resources/Scripting/NetStandard/ref/2.1.0"
UE="$U/Resources/Scripting/Managed/UnityEngine"
SA="Library/ScriptAssemblies"

for required in "$DOTNET" "$CSC" "$NETSTD/netstandard.dll"; do
    [ -e "$required" ] || { echo "MISSING: $required"; exit 1; }
done

REFS=("-r:$NETSTD/netstandard.dll")
for f in "$NETSTD"/*.dll;       do REFS+=("-r:$f"); done
for f in "$UE"/UnityEngine*.dll; do REFS+=("-r:$f"); done
for f in "$U/Resources/Scripting/Managed"/UnityEditor*.dll; do REFS+=("-r:$f"); done
# Package assemblies Unity has already built (ML-Agents, UI, TextMeshPro, ...).
for f in "$SA"/*.dll; do
    case "$(basename "$f")" in Assembly-CSharp*.dll) continue ;; esac
    REFS+=("-r:$f")
done

SRC=(Assets/Core/Environment/*.cs Assets/Core/ML/*.cs Assets/Core/Physics/*.cs
     Assets/Core/Presentation/*.cs
     Assets/UI/*.cs Assets/Audio/SFX/*.cs Assets/Editor/*.cs)

OUT="$(mktemp -d)/check.dll"
LOG="$(mktemp)"

"$DOTNET" "$CSC" -nologo -target:library -langversion:9 -nostdlib+ \
    -nowarn:0169,0414,0649,0108 \
    "${REFS[@]}" -out:"$OUT" "${SRC[@]}" >"$LOG" 2>&1
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "COMPILES CLEAN"
    exit 0
fi

echo "COMPILE FAILED (exit $STATUS)"
echo
grep -E ": error " "$LOG" | head -40
exit 1
