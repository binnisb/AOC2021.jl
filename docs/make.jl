using AOC2021
using Documenter

DocMeta.setdocmeta!(AOC2021, :DocTestSetup, :(using AOC2021); recursive=true)

makedocs(;
    modules=[AOC2021],
    authors="Brynjar Smári Bjarnason <binni@binnisb.com> and contributors",
    repo="https://github.com/binnisb/AOC2021.jl/blob/{commit}{path}#{line}",
    sitename="AOC2021.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://binnisb.github.io/AOC2021.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/binnisb/AOC2021.jl",
    devbranch="main",
)
