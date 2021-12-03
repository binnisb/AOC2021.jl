using AOC2021
using Test

@testset "AOC2021.jl" begin
    # Write your tests here.
    @testset "Test Solutions" begin
        @test solve(TestInput, 1, 1) == 7
        @test solve(TestInput, 1, 2) == 5

        @test solve(TestInput, 2, 1) == 150
        @test solve(TestInput, 2, 2) == 900

        @test solve(TestInput, 3, 1) == 198
        @test solve(TestInput, 3, 2) == 230

    end

    @testset "Real Solutions" begin
        @test solve(FullInput, 1, 1) == 1581
        @test solve(FullInput, 1, 2) == 1618
        @test solve(FullInput, 2, 1) == 1746616
        @test solve(FullInput, 2, 2) == 1741971043
        @test solve(FullInput, 3, 1) == 1997414
        @test solve(FullInput, 3, 2) == 1032597
    end
end
