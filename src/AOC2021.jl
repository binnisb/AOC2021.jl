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
    read(::Type{FullInput}, num::Int) = read(Puzzle, "$(@__DIR__)/../data/puzzle/$(num).txt")
    read(::Type{TestInput}, num::Int) = read(Puzzle, "$(@__DIR__)/../data/puzzle_test/$(num).txt")


    solve(::Type{T}, number, part; only_data=false) where T <: Puzzle = read(T,number) |> input-> solve(Val(number), input) |> x-> only_data ? (return x) : solve( Val(number), Val(part), x)
    
    
    solve(::Val{1}, input) = shift -> map(x->parse(Int,x), input) |> nums->
        sum(1 for i in (shift+1):length(nums) if nums[i] > nums[i-shift])

    solve(::Val{1}, ::Val{1}, shift) = shift(1)
    solve(::Val{1}, ::Val{2}, shift) = shift(3)

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
    solve(::Val{2}, ::Val{1}, directions) =  directions.forward*directions.aim
    solve(::Val{2}, ::Val{2}, directions) =  directions.forward*directions.depth

    solve(::Val{3}, input) = input .|> collect .|> (x->parse.(Int,x)) |> x-> hcat(x...) |> transpose

    sol_3_final_processing(vals) = vals .|>
        (x-> [c > 0 ? '1' : '0' for c in x]) .|>
        join |>
        x-> parse.(Int, x; base=2) |>
        x->reduce(*,x)

    solve(::Val{3}, ::Val{1}, matr) = matr |>
        x->sum(x;dims=1) |>
        x-> (x.>(size(matr)[1]/2)) |>
        x-> [x, .!x] |>
        sol_3_final_processing
    
    solve(::Val{3}, ::Val{2}, matr) = begin
        rows, cols = size(matr)
        ind_selector(m, range, c, select_ones) = begin
            r = length(range)
            if r == 1
                return m[range,:]
            else
                s = sum(m[range,c])
                range = select_ones(s,range) ? (first(range):(first(range) + s-1)) : (first(range)+s):last(range)
                ind_selector(m, range, c+1, select_ones)
            end
        end

        matr |>
            m->sortslices(m, dims=1, rev=true) |>
            m->[ind_selector(m, 1:rows, 1, (s,r)-> 2*s >= length(r)), ind_selector(m, 1:rows, 1, (s,r)-> 2*s < length(r))] |>
            sol_3_final_processing
    end
end
