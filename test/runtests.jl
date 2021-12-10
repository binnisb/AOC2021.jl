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

        @test solve(TestInput, 8, 1) == 26
        @test solve(TestInput, 8, 2) == 61229

        @test solve(TestInput, 9, 1) == 15
        @test solve(TestInput, 9, 2) == 1134

        @test solve(TestInput, 10, 1) == 26397
        @test solve(TestInput, 10, 2) == 288957
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

        @test solve(FullInput, 7, 1) == 339321
        @test solve(FullInput, 7, 2) == 95476244

        @test solve(FullInput, 8, 1) == 476
        @test solve(FullInput, 8, 2) == 1011823

        @test solve(FullInput, 9, 1) == 452
        @test solve(FullInput, 9, 2) == 1263735
    end
end
