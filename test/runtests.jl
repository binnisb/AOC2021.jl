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

        @test solve(TestInput, 4, 1) == 4512
        @test solve(TestInput, 4, 2) == 1924

        @test solve(TestInput, 5, 1) == 5
        @test solve(TestInput, 5, 2) == 12

        @test solve(TestInput, 6, 1; days=18) == 26
        @test solve(TestInput, 6, 1) == 5934
        
        @test solve(TestInput, 6, 2; days=256) == 26984457539
        
        @test solve(TestInput, 7, 1) == 37
        @test solve(TestInput, 7, 2) == 168
    end

    @testset "Real Solutions" begin
        @test solve(FullInput, 1, 1) == 1581
        @test solve(FullInput, 1, 2) == 1618
        @test solve(FullInput, 2, 1) == 1746616
        @test solve(FullInput, 2, 2) == 1741971043
        @test solve(FullInput, 3, 1) == 1997414
        @test solve(FullInput, 3, 2) == 1032597
        @test solve(FullInput, 4, 1) == 28082
        @test solve(FullInput, 4, 2) == 8224
        @test solve(FullInput, 5, 1) == 7674
        @test solve(FullInput, 6, 1; days=80) == 380758
        @test solve(FullInput, 6, 2; days=256) == 1710623015163
    end
end
