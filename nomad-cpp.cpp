#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <fstream>
#include <cstdint>

// --- Constants ---
constexpr int S = 12;           // Sub-quantizers
constexpr int n_centroids = 16;  
constexpr int d_sub = 64;        
constexpr int n_keys = 512;      // Context length

// Quantized LUT and Centroids 
alignas(32) uint8_t LUT[S][n_centroids];  // 8-bit quantized
alignas(32) float centroids[S][n_centroids][d_sub];

// --- SIMD Lookup Kernel ---
void nomad_attention(const uint8_t* key_codes, float* scores) {
    for (int s = 0; s < S; ++s) {
        __m256i lut = _mm256_load_si256((__m256i*)LUT[s]);
        for (int i = 0; i < n_keys; i += 16) {
            // Load 16x 4-bit codes (packed as 8-bit)
            __m128i codes = _mm_loadu_si128((__m128i*)(key_codes + s * n_keys / 2 + i / 2));
            
            // Split into lower/upper 4-bit nibbles
            __m128i lo = _mm_and_si128(codes, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_srli_epi16(codes, 4);
            
            // Lookup with AVX2
            __m256i res_lo = _mm256_shuffle_epi8(lut, _mm256_cvtepu8_epi32(lo));
            __m256i res_hi = _mm256_shuffle_epi8(lut, _mm256_cvtepu8_epi32(hi));
            
            // Accumulate
            __m256i sum = _mm256_add_epi32(res_lo, res_hi);
            _mm256_storeu_ps(scores + i, _mm256_cvtepi32_ps(sum));
        }
    }
}

// --- Benchmark ---
int main() {
    // Initialize mock data
    std::vector<uint8_t> key_codes(S * n_keys / 2, 0);  // 4-bit packed
    std::vector<float> scores(n_keys, 0.0f);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        nomad_attention(key_codes.data(), scores.data());
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    const int trials = 1000;
    for (int i = 0; i < trials; ++i) {
        nomad_attention(key_codes.data(), scores.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    double avg_time = std::chrono::duration<double>(end - start).count() / trials * 1e6;

    std::cout << "Avg NoMAD-Attention time: " << avg_time << " μs" << std::endl;
    return 0;
}