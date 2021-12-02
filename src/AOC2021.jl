module AOC2021
    export solve, FullInput, TestInput
# Write your package code here.
    
    abstract type Puzzle end
    abstract type FullInput <: Puzzle end
    abstract type TestInput <: Puzzle end
    
    read(::Type{Puzzle}, path::String) = begin
        open(path) do f
            readlines(f)
        end
    end
    read(::Type{FullInput}, num::Int) = read(Puzzle, "/home/binni/projects/AdventOfCode2021/data/exp_raw/puzzle/$(num).txt")
    read(::Type{TestInput}, num::Int) = read(Puzzle, "/home/binni/projects/AdventOfCode2021/data/exp_raw/puzzle_test/$(num).txt")


    solve(::Type{T}, number, part) where T <: Puzzle = read(T,number) |> x->solve( Val(number), Val(part), x)
    
    
    solve(::Val{1}, input, shift=1) = map(x->parse(Int,x), input) |> nums->
        sum(1 for i in (shift+1):length(nums) if nums[i] > nums[i-shift])

    solve(::Val{1}, ::Val{1}, input) = solve(Val(1),input, 1)
    solve(::Val{1}, ::Val{2}, input) = solve(Val(1),input, 3)

    solve(::Val{2}, input) = begin
        global forward = 0
        global depth = 0
        global aim = 0
        for l in input
            d, v = split(l)
            v = parse(Int,v)
            if d == "forward"
                forward += v
                depth += v*aim
            elseif d == "down"
                aim += v
            else
                aim -= v
            end
        end
        (forward=forward, depth=depth, aim=aim)
    end
    solve(::Val{2}, ::Val{1}, input) = solve(Val(2), input) |> x-> x.forward*x.aim
    solve(::Val{2}, ::Val{2}, input) = solve(Val(2), input) |> x-> x.forward*x.depth
end
