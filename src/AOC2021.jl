module AOC2021
    using Memoize: @memoize
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


    solve(::Type{T}, number, part; only_data=false, kwargs...) where T <: Puzzle = read(T,number) |> input-> solve(Val(number), input) |> x-> only_data ? (return x) : solve( Val(number), Val(part);input=x, kwargs...)
    
    
    solve(::Val{1}, input) = shift -> map(x->parse(Int,x), input) |> nums->
        sum(1 for i in (shift+1):length(nums) if nums[i] > nums[i-shift])

    solve(::Val{1}, ::Val{1}; input) = input(1)
    solve(::Val{1}, ::Val{2}; input) = input(3)

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
    solve(::Val{2}, ::Val{1}; input) =  input.forward*input.aim
    solve(::Val{2}, ::Val{2}; input) =  input.forward*input.depth

    solve(::Val{3}, input) = input .|> collect .|> (x->parse.(Int,x)) |> x-> hcat(x...) |> transpose

    sol_3_final_processing(vals) = vals .|>
        (x-> [c > 0 ? '1' : '0' for c in x]) .|>
        join |>
        x-> parse.(Int, x; base=2) |>
        x->reduce(*,x)

    solve(::Val{3}, ::Val{1}; input) = input |>
        x->sum(x;dims=1) |>
        x-> (x.>(size(input)[1]/2)) |>
        x-> [x, .!x] |>
        sol_3_final_processing
    
    solve(::Val{3}, ::Val{2}; input) = begin
        matr=input
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

    struct Board
        b::Matrix{Int}
        found::Matrix{Bool}
        Board(lines) = begin
            board = lines .|> split .|> (x-> parse.(Int,x)) |> x-> hcat(x...)
            new(board, zeros(Bool, size(board)))
        end
    end
    bingo!(n::Int, b::Board) = begin
        b.found[b.b .== n] .= true
        max(sum(b.found;dims=1)...,sum(b.found;dims=2)...) == 5 ?  sum(b.b[.!b.found])*n : nothing
    end

    bingos!(nums::Vector{Int}, boards::Vector{Board}; get_first=true) = begin
        for n in nums
            boards_result = bingo!.(n,boards)
            nothings = isnothing.(boards_result)
            if get_first && !all(nothings)
                res = boards_result[.!nothings]
                length(res) == 1 || throw(DomainError("Too many?"))
                return res[1]
            elseif !get_first
                if all(.!nothings)
                    length(boards_result) == 1 || throw(DomainError("TooMany?"))
                    return boards_result[1]
                else
                    boards = boards[nothings]
                end
            end
        end
    end
    solve(::Val{4}, input) = (nums=parse.(Int,split(first(input),',')), boards=[Board(input[(3:7).+((i-1)*6)]) for i in 1:div(length(input),6)])
    solve(::Val{4}, ::Val{1}; input) = bingos!(input[1], input[2])
    solve(::Val{4}, ::Val{2}; input) =  bingos!(input[1], input[2]; get_first=false)

    struct Grid
        g::Matrix{Int}
        isdiag::Bool
        Grid(line) = begin
            ps = line |> x-> map(strip, split(x, "->")) |> ps -> split.(ps,",") .|> ps -> parse.(Int,ps)
            x1 = ps[1][1]
            x2 = ps[2][1]
            y1 = ps[1][2]
            y2 = ps[2][2]
            max_len = max(abs(x2-x1),abs(y2-y1))

            s(x) = x == 0 ? 1 : sign(x)
            r(a1,a2) = a1 == a2 ? [a1 for _ in 1:max_len+1] : a1:s(a2-a1):a2
            m = zeros(Int,max(x1,x2)+1,max(y1,y2)+1)
            for p in zip(r(x1,x2), r(y1,y2))
                m[p[1]+1,p[2]+1] = 1
            end
            new(m, x1==x2 || y1 == y2)
        end
    end
    data5(line) = begin
        ps = line |> x-> map(strip, split(x, "->")) |> ps -> split.(ps,",") .|> ps -> parse.(Int,ps)
        x1 = ps[1][1]
        x2 = ps[2][1]
        y1 = ps[1][2]
        y2 = ps[2][2]
        max_len = max(abs(x2-x1),abs(y2-y1))

        s(x) = x == 0 ? 1 : sign(x)
        r(a1,a2) = a1 == a2 ? [a1 for _ in 1:max_len+1] : a1:s(a2-a1):a2

        (x1==x2 || y1 == y2, Dict((p[1]+1,p[2]+1) => 1 for p in zip(r(x1,x2), r(y1,y2))))
    end
    solve(::Val{5}, input) = data5.(input)
    solve(::Val{5}, ::Val{1}; input, use_diag=false) = begin
        grids = use_diag ? [x[2] for x in input] : [x[2] for x in input if x[1]]
        mergewith(+,grids...) |> values |> (x-> x.>1) |> sum
    end

    solve(::Val{5}, ::Val{2}; input) = solve(Val(5), Val(1); input, use_diag=true)

    solve(::Val{6}, lines) = lines |> 
        l->(length(l) == 1 ? l[1] : throw(DomainError(l, "Should only be one line"))) |>
        s-> split(s,",") |>
        n->parse.(Int,n)

    """ Got some help. I solved 6.1 but 6.2 was too deeply nested so I built the cache first by hand
    which was a bit harder than I had hoped. Saw a solution using the @memoize function, and what a difference.
    Hides so much boilerplate and "JustWorks"
    """
    @memoize fish(counter, days) = begin
        days == 0 && return 1
        counter == 0 && return fish(8,days-1) + fish(6,days-1)
        return fish(counter-1, days-1)
    end

    solve(::Val{6}, ::Val{1}; input, days=80) = begin
        # input |> counter |> (d->[v*fish(k, days) for (k,v) in d]) |> sum
        # No need for the counter since fishes are memoized
        fish1(k) = fish(k,days)
        input .|> fish1 |> sum
    end
    solve(::Val{6}, ::Val{2}; input, days=256) = solve(Val(6), Val(1); input=input,days=days)

    solve(::Val{7}, lines) = parse.(Int, split(lines[1],","))
    solve(::Val{7},::Val{1}; input, red_func=identity) = begin
        global res = 100000000
        for i in UnitRange(extrema(input)...)
            lens = abs.(i.-input)
            s = sum([red_func(l) for l in lens])
            s < res && (res = s)
        end
        res
    end
    solve(::Val{7},::Val{2}; input) = solve(Val(7),Val(1); input=input, red_func=(x->convert(Int,(x*(x+1)/2))))
    
    solve(::Val{8}, lines) = lines .|> (l -> split(l,"|") .|> strip .|> split)
    count_uniqs_8(res_nums) = res_nums .|> length .|> (x-> x ∈ [2,4, 3, 7]) |> sum
    solve(::Val{8}, ::Val{1}; input) = input .|> (x->count_uniqs_8(x[2])) |> sum
    parse_input_decode_output(nums, vals) = begin
        nums = Set.(nums)
        num_to_str = Dict{Int,Set{Char}}()
        decode = Dict{Char, Char}()

        lengths = length.(nums)

        for (i,(l,n)) in enumerate(zip(lengths,nums))
            l == 2 && (num_to_str[1] = n)
            l == 3 && (num_to_str[7] = n)
            l == 4 && (num_to_str[4] = n)
            l == 7 && (num_to_str[8] = n)
        end
        eight = num_to_str[8]
        decode[setdiff(num_to_str[7],num_to_str[1]) |> pop!] = 'a'
        eg = setdiff(eight,num_to_str[4])
        for c in eg
            if setdiff(eight,c) in nums
                decode[c] = 'e'
                num_to_str[9] = setdiff(eight, c)
            else
                decode[c] = 'g'
            end
        end

        bd = setdiff(num_to_str[4],num_to_str[1])
        for c in bd
            if setdiff(eight,c) in nums
                decode[c] = 'd'
                num_to_str[0] = setdiff(eight, c)
            else
                decode[c] = 'b'
            end
        end
        nums = setdiff(nums,values(num_to_str))
        num_to_str[6] = [n for n in nums if length(n) == 6][1]
        decode[setdiff(eight, num_to_str[6]) |> pop!] = 'c'
        decode[setdiff('a':'g',keys(decode)) |> pop!] = 'f'

        encode = Dict((v,k) for (k,v) in decode)
        num_to_str[3] = setdiff(setdiff(eight,encode['b']),encode['e'])
        num_to_str[5] = setdiff(num_to_str[6], encode['e'])
        num_to_str[2] = setdiff(setdiff(eight,encode['b']),encode['f'])

        str_to_num = Dict((v,k) for (k,v) in num_to_str)
        
        Set.(vals) .|> (s-> str_to_num[s]) |> join |> x->parse(Int,x)
    end 
    solve(::Val{8}, ::Val{2}; input) = input .|> (x->bool_decode(x...)) |> sum

    bool_decode(nums, vals) = begin
        s1(x) = sum(x,dims=1) |> vec
        model = 'a':'g'
        bm = hcat((nums .|> (n -> model .∈ n))...)
        vm = hcat((vals .|> (n -> model .∈ n))...)
        s = sum(bm, dims=1)

        # Index of number in bm
        found = Dict(
            i=>findfirst(isequal(c),s)[2] for (i,c) in zip([1,4,7,8],[2,4,3,7])
        )

        bf(n) = bm[:,found[n]]
        bi(ns) = bm[:,ns]

        ind_069 = (bm .⊻ bf(8)) |> s1 |> x->findall(isequal(1),x)
        found[6] = bi(ind_069) .| bf(7) |> s1 .|> isequal(7) |> x-> popat!(ind_069, findfirst(x))
        found[0] = bi(ind_069) .| bf(4) |> s1 .|> isequal(7) |> x-> popat!(ind_069, findfirst(x))
        found[9] = ind_069[1]

        ind_235 = filter(x->x ∉ values(found), 1:10)
        found[2] = bi(ind_235) .| bf(4) |> s1 .|> isequal(7) |> x->popat!(ind_235, findfirst(x))
        found[5] = bi(ind_235) .| bf(2) |> s1 .|> isequal(7) |> x->popat!(ind_235, findfirst(x))
        found[3] = ind_235[1]
        new_b = bi([found[k] for k in 0:9])
        decode = Dict(v=>k for (k,v) in found)
        eachcol(vm) .|> (x-> x.==bm) .|> (x->all(x,dims=1)) .|> vec .|> findfirst .|> (x->decode[x]) |> join |> x->parse(Int,x)
        
    end
    pints(x) = parse.(Int,x)
    solve(::Val{9}, lines) = begin
        input = lines .|> collect .|> pints |> x->hcat(x...) 
        r,c = size(input)
        m = zeros(Int, r+2,c+2)
        ma = maximum(input) + 10000
        m[:,1] .= ma
        m[:,end] .= ma
        m[1,:] .= ma
        m[end,:] .= ma

        m[2:end-1,2:end-1] = input[:]
        m
    end
    get_low_points(m) = begin
        r,c = size(m)
        check(i,j) =  (([m[i,j-1], m[i,j+1], m[i,j], m[i-1,j], m[i+1,j]] .- m[i,j]) .> 0) |> sum |> isequal(4)
        res = zeros(Bool,r,c)
        res[2:end-1, 2:end-1] .= [check(i,j) for i in 2:r-1, j in 2:c-1]
        res
    end
    solve(::Val{9}, ::Val{1}; input) = input |> get_low_points |> x->sum(input[x].+1)
    valley_size_9(c,valleys) = begin
        shifts = [CartesianIndex(0,1), CartesianIndex(0,-1), CartesianIndex(1,0), CartesianIndex(-1,0)]
        to_test = Set{CartesianIndex{2}}([c])
        done = Set{CartesianIndex{2}}()
        global count = 0
        while !isempty(to_test)
            t = pop!(to_test)
            push!(done,t)
            count += valleys[t]
            new_points = setdiff([t+s for s in shifts if valleys[t+s] != 0], done)
            !isempty(new_points) && push!(to_test, new_points...)
        end
        count

    end
    solve(::Val{9}, ::Val{2}; input) = begin
        m = get_low_points(input)
        valleys = input .< 9
        vs(x) = valley_size_9(x,valleys)
        carts = m |> findall .|> vs |> sort |> reverse |> x->reduce(*,x[1:3])

    end

end
