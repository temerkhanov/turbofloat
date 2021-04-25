// Turbofloat.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <vector>
#include <random>
#include <sstream>
#include <limits>
#include <chrono>

#include <intrin.h>
#include <immintrin.h>

typedef __m256i simd_m256i;
typedef __m128i simd_m128i;
typedef __m64   simd_m64i;

#define simd_mm256_set1_epi8         _mm256_set1_epi8
#define simd_mm256_set1_epi32        _mm256_set1_epi32
#define simd_mm256_set_epi32         _mm256_set_epi32

#define simd_mm256_extract_epi8      _mm256_extract_epi8
#define simd_mm256_extracti128_si256 _mm256_extracti128_si256
#define simd_mm256_extract_epi64     _mm256_extract_epi64

#define simd_mm256_loadu_si256       _mm256_loadu_si256
#define simd_mm256_maskload_epi32    _mm256_maskload_epi32

#define simd_mm256_cmpgt_epi8        _mm256_cmpgt_epi8
#define simd_mm256_movemask_epi8     _mm256_movemask_epi8
#define simd_mm256_cmpgt_epi32       _mm256_cmpgt_epi32
#define simd_mm256_movemask_epi32    _mm256_movemask_epi32

#define simd_mm256_or_si256          _mm256_or_si256

//----------------------------------------------------
#define simd_mm_set1_epi8            _mm_set1_epi8
#define simd_mm_setr_epi8            _mm_setr_epi8
#define simd_mm_set_epi8             _mm_set_epi8

#define simd_mm_set1_epi16           _mm_set1_epi16
#define simd_mm_setr_epi16           _mm_setr_epi16
#define simd_mm_set_epi16            _mm_set_epi16

#define simd_mm_set_epi32            _mm_set_epi32

#define simd_mm_set_epi64x           _mm_set_epi64x

#define simd_mm_loadu_si32           _mm_loadu_si32
#define simd_mm_load_si128           _mm_load_si128
#define simd_mm_loadu_si128          _mm_loadu_si128

#define simd_mm_sllv_epi64           _mm_sllv_epi64

#define simd_mm_extract_epi64        _mm_extract_epi64
#define simd_mm_extract_epi16        _mm_extract_epi16
#define simd_mm_extract_epi8         _mm_extract_epi8

#define simd_mm_shuffle_epi8         _mm_shuffle_epi8
#define simd_mm_shufflelo_epi16      _mm_shufflelo_epi16
#define simd_mm_shufflehi_epi16      _mm_shufflehi_epi16

#define simd_mm_bsrli_si128          _mm_bsrli_si128
#define simd_mm_bslli_si128          _mm_bslli_si128
#define simd_mm_bslli_epi64          _mm_bslli_epi64
#define simd_mm_bsrli_epi64          _mm_bsrli_epi64
#define simd_mm_slli_epi64           _mm_slli_epi64
#define simd_mm_srli_epi64           _mm_srli_epi64
#define simd_mm_slli_epi32           _mm_slli_epi32
#define simd_mm_srli_epi32           _mm_srli_epi32

#define simd_mm_add_epi64            _mm_add_epi64

#define simd_mm_subs_epu8            _mm_subs_epi8
#define simd_mm_adds_epu8            _mm_adds_epu8
#define simd_mm_sub_epi8             _mm_sub_epi8
#define simd_mm_add_epi8             _mm_add_epi8

#define simd_mm_mul_epu32            _mm_mul_epu32

#define simd_mm_maddubs_epi16        _mm_maddubs_epi16
#define simd_mm_madd_epi16           _mm_madd_epi16

#define simd_mm_or_si128             _mm_or_si128
#define simd_mm_xor_si128            _mm_xor_si128
#define simd_mm_andnot_si128         _mm_andnot_si128

#define simd_mm_cmpgt_epi8           _mm_cmpgt_epi8

#define bits_tzcnt_u32               _tzcnt_u32
#define bits_bittest                 _bittest
#define bits_bitreset                _bittestandreset

inline
bool
is_integer(int8_t x) noexcept
{
    return ((x >= '0') && (x <= '9'));
}

inline
bool is_dot(char x) noexcept
{
    return (x == '.');
}

inline
bool is_exp(char x) noexcept
{
    return (x == 'e') || (x == 'E');
}

inline
bool is_sign(char x) noexcept
{
    return (x == '-') || (x == '+');
}

inline
uint32_t str_get_bitmask(const char * __restrict str) noexcept
{
    simd_m256i x = simd_mm256_loadu_si256((simd_m256i *)str);

    simd_m256i zeros = simd_mm256_set1_epi8('0');
    simd_m256i nines = simd_mm256_set1_epi8('9');

    simd_m256i y = simd_mm256_cmpgt_epi8(zeros, x);
    simd_m256i z = simd_mm256_cmpgt_epi8(x, nines);

    x = simd_mm256_or_si256(y, z);

    return simd_mm256_movemask_epi8(x);
}

inline
simd_m256i str_load_m(const char* str_in, size_t len) noexcept
{
    uint32_t bytes = len + 1; // with 0 delimiter
    simd_m256i mask0, mask1, mask, str;

    mask0 = simd_mm256_set_epi32(7 * sizeof(uint32_t), 6 * sizeof(uint32_t),
                                  5 * sizeof(uint32_t), 4 * sizeof(uint32_t),
                                  3 * sizeof(uint32_t), 2 * sizeof(uint32_t),
                                  1 * sizeof(uint32_t), 0 * sizeof(uint32_t));
    mask1 = simd_mm256_set1_epi32(bytes);
    mask  = simd_mm256_cmpgt_epi32(mask1, mask0); // which 32-bit words to load

    str = simd_mm256_maskload_epi32((int32_t *)str_in, mask);

    return str;
}

inline
simd_m256i str_load_256b(const char* str_in) noexcept
{
    simd_m256i str;
    str = simd_mm256_loadu_si256((simd_m256i *) str_in);

    return str;
}

inline
simd_m128i str_load_128b(const char* str_in) noexcept
{
    simd_m128i str;
    str = simd_mm_loadu_si128((simd_m128i*)str_in);

    return str;
}

inline
uint64_t str_load_64b(const char* str_in) noexcept
{
    uint64_t str;
    str = *((uint64_t*)str_in);

    return str;
}

inline
uint32_t str_load_32b(const char* str_in) noexcept
{
    uint32_t str;
    str = *((uint32_t*)str_in);

    return str;
}

inline
uint16_t str_load_16b(const char* str_in) noexcept
{
    uint16_t str;
    str = *((uint16_t*)str_in);

    return str;
}

inline
uint32_t set_mask_zero(uint32_t x, size_t index) noexcept
{
    return (x & ~(1 << index));
}

inline
bool bit_is_set(uint32_t x, size_t index) noexcept
{
    return (x & (1 << index));
}

inline
bool bit_is_zero(uint32_t x, size_t index) noexcept
{
    return !(x & (1 << index));
}

inline
uint32_t pos_non_zero(uint32_t x) noexcept
{
    unsigned long ret = 0;
    ret = bits_tzcnt_u32(x);
    return ret;
}

inline
uint8_t str_extract_byte(const char *str, size_t index) noexcept
{
    return str[index];
}

const double decimals_d[] = {
    1.0e308,
    1.0e307, 1.0e306, 1.0e305, 1.0e304,
    1.0e303, 1.0e302, 1.0e301, 1.0e300,
    1.0e299, 1.0e298, 1.0e297, 1.0e296,
    1.0e295, 1.0e294, 1.0e293, 1.0e292,
    1.0e291, 1.0e290, 1.0e289, 1.0e288,
    1.0e287, 1.0e286, 1.0e285, 1.0e284,
    1.0e283, 1.0e282, 1.0e281, 1.0e280,
    1.0e279, 1.0e278, 1.0e277, 1.0e276,
    1.0e275, 1.0e274, 1.0e273, 1.0e272,
    1.0e271, 1.0e270, 1.0e269, 1.0e268,
    1.0e267, 1.0e266, 1.0e265, 1.0e264,
    1.0e263, 1.0e262, 1.0e261, 1.0e260,
    1.0e259, 1.0e258, 1.0e257, 1.0e256,
    1.0e255, 1.0e254, 1.0e253, 1.0e252,
    1.0e251, 1.0e250, 1.0e249, 1.0e248,
    1.0e247, 1.0e246, 1.0e245, 1.0e244,
    1.0e243, 1.0e242, 1.0e241, 1.0e240,
    1.0e239, 1.0e238, 1.0e237, 1.0e236,
    1.0e235, 1.0e234, 1.0e233, 1.0e232,
    1.0e231, 1.0e230, 1.0e229, 1.0e228,
    1.0e227, 1.0e226, 1.0e225, 1.0e224,
    1.0e223, 1.0e222, 1.0e221, 1.0e220,
    1.0e219, 1.0e218, 1.0e217, 1.0e216,
    1.0e215, 1.0e214, 1.0e213, 1.0e212,
    1.0e211, 1.0e210, 1.0e209, 1.0e208,
    1.0e207, 1.0e206, 1.0e205, 1.0e204,
    1.0e203, 1.0e202, 1.0e201, 1.0e200,
    1.0e199, 1.0e198, 1.0e197, 1.0e196,
    1.0e195, 1.0e194, 1.0e193, 1.0e192,
    1.0e191, 1.0e190, 1.0e189, 1.0e188,
    1.0e187, 1.0e186, 1.0e185, 1.0e184,
    1.0e183, 1.0e182, 1.0e181, 1.0e180,
    1.0e179, 1.0e178, 1.0e177, 1.0e176,
    1.0e175, 1.0e174, 1.0e173, 1.0e172,
    1.0e171, 1.0e170, 1.0e169, 1.0e168,
    1.0e167, 1.0e166, 1.0e165, 1.0e164,
    1.0e163, 1.0e162, 1.0e161, 1.0e160,
    1.0e159, 1.0e158, 1.0e157, 1.0e156,
    1.0e155, 1.0e154, 1.0e153, 1.0e152,
    1.0e151, 1.0e150, 1.0e149, 1.0e148,
    1.0e147, 1.0e146, 1.0e145, 1.0e144,
    1.0e143, 1.0e142, 1.0e141, 1.0e140,
    1.0e139, 1.0e138, 1.0e137, 1.0e136,
    1.0e135, 1.0e134, 1.0e133, 1.0e132,
    1.0e131, 1.0e130, 1.0e129, 1.0e128,
    1.0e127, 1.0e126, 1.0e125, 1.0e124,
    1.0e123, 1.0e122, 1.0e121, 1.0e120,
    1.0e119, 1.0e118, 1.0e117, 1.0e116,
    1.0e115, 1.0e114, 1.0e113, 1.0e112,
    1.0e111, 1.0e110, 1.0e109, 1.0e108,
    1.0e107, 1.0e106, 1.0e105, 1.0e104,
    1.0e103, 1.0e102, 1.0e101, 1.0e100,
    1.0e099, 1.0e098, 1.0e097, 1.0e096,
    1.0e095, 1.0e094, 1.0e093, 1.0e092,
    1.0e091, 1.0e090, 1.0e089, 1.0e088,
    1.0e087, 1.0e086, 1.0e085, 1.0e084,
    1.0e083, 1.0e082, 1.0e081, 1.0e080,
    1.0e079, 1.0e078, 1.0e077, 1.0e076,
    1.0e075, 1.0e074, 1.0e073, 1.0e072,
    1.0e071, 1.0e070, 1.0e069, 1.0e068,
    1.0e067, 1.0e066, 1.0e065, 1.0e064,
    1.0e063, 1.0e062, 1.0e061, 1.0e060,
    1.0e059, 1.0e058, 1.0e057, 1.0e056,
    1.0e055, 1.0e054, 1.0e053, 1.0e052,
    1.0e051, 1.0e050, 1.0e049, 1.0e048,
    1.0e047, 1.0e046, 1.0e045, 1.0e044,
    1.0e043, 1.0e042, 1.0e041, 1.0e040,
    1.0e039, 1.0e038, 1.0e037, 1.0e036,
    1.0e035, 1.0e034, 1.0e033, 1.0e032,
    1.0e031, 1.0e030, 1.0e029, 1.0e028,
    1.0e027, 1.0e026, 1.0e025, 1.0e024,
    1.0e023, 1.0e022, 1.0e021, 1.0e020,
    1.0e019, 1.0e018, 1.0e017, 1.0e016,
    1.0e015, 1.0e014, 1.0e013, 1.0e012,
    1.0e011, 1.0e010, 1.0e009, 1.0e008,
    1.0e007, 1.0e006, 1.0e005, 1.0e004,
    1.0e003, 1.0e002, 1.0e001, 1.0e000,
    //Negative part
              1.0e-001, 1.0e-002, 1.0e-003,
    1.0e-004, 1.0e-005, 1.0e-006, 1.0e-007,
    1.0e-008, 1.0e-009, 1.0e-010, 1.0e-011,
    1.0e-012, 1.0e-013, 1.0e-014, 1.0e-015,
    1.0e-016, 1.0e-017, 1.0e-018, 1.0e-019,
    1.0e-020, 1.0e-021, 1.0e-022, 1.0e-023,
    1.0e-024, 1.0e-025, 1.0e-026, 1.0e-027,
    1.0e-028, 1.0e-029, 1.0e-030, 1.0e-031,
    1.0e-032, 1.0e-033, 1.0e-034, 1.0e-035,
    1.0e-036, 1.0e-037, 1.0e-038, 1.0e-039,
    1.0e-040, 1.0e-041, 1.0e-042, 1.0e-043,
    1.0e-044, 1.0e-045, 1.0e-046, 1.0e-047,
    1.0e-048, 1.0e-049, 1.0e-050, 1.0e-051,
    1.0e-052, 1.0e-053, 1.0e-054, 1.0e-055,
    1.0e-056, 1.0e-057, 1.0e-058, 1.0e-059,
    1.0e-060, 1.0e-061, 1.0e-062, 1.0e-063,
    1.0e-064, 1.0e-065, 1.0e-066, 1.0e-067,
    1.0e-068, 1.0e-069, 1.0e-070, 1.0e-071,
    1.0e-072, 1.0e-073, 1.0e-074, 1.0e-075,
    1.0e-076, 1.0e-077, 1.0e-078, 1.0e-079,
    1.0e-080, 1.0e-081, 1.0e-082, 1.0e-083,
    1.0e-084, 1.0e-085, 1.0e-086, 1.0e-087,
    1.0e-088, 1.0e-089, 1.0e-090, 1.0e-091,
    1.0e-092, 1.0e-093, 1.0e-094, 1.0e-095,
    1.0e-096, 1.0e-097, 1.0e-098, 1.0e-099,
    1.0e-100, 1.0e-101, 1.0e-102, 1.0e-103,
    1.0e-104, 1.0e-105, 1.0e-106, 1.0e-107,
    1.0e-108, 1.0e-109, 1.0e-110, 1.0e-111,
    1.0e-112, 1.0e-113, 1.0e-114, 1.0e-115,
    1.0e-116, 1.0e-117, 1.0e-118, 1.0e-119,
    1.0e-120, 1.0e-121, 1.0e-122, 1.0e-123,
    1.0e-124, 1.0e-125, 1.0e-126, 1.0e-127,
    1.0e-128, 1.0e-129, 1.0e-130, 1.0e-131,
    1.0e-132, 1.0e-133, 1.0e-134, 1.0e-135,
    1.0e-136, 1.0e-137, 1.0e-138, 1.0e-139,
    1.0e-140, 1.0e-141, 1.0e-142, 1.0e-143,
    1.0e-144, 1.0e-145, 1.0e-146, 1.0e-147,
    1.0e-148, 1.0e-149, 1.0e-150, 1.0e-151,
    1.0e-152, 1.0e-153, 1.0e-154, 1.0e-155,
    1.0e-156, 1.0e-157, 1.0e-158, 1.0e-159,
    1.0e-160, 1.0e-161, 1.0e-162, 1.0e-163,
    1.0e-164, 1.0e-165, 1.0e-166, 1.0e-167,
    1.0e-168, 1.0e-169, 1.0e-170, 1.0e-171,
    1.0e-172, 1.0e-173, 1.0e-174, 1.0e-175,
    1.0e-176, 1.0e-177, 1.0e-178, 1.0e-179,
    1.0e-180, 1.0e-181, 1.0e-182, 1.0e-183,
    1.0e-184, 1.0e-185, 1.0e-186, 1.0e-187,
    1.0e-188, 1.0e-189, 1.0e-190, 1.0e-191,
    1.0e-192, 1.0e-193, 1.0e-194, 1.0e-195,
    1.0e-196, 1.0e-197, 1.0e-198, 1.0e-199,
    1.0e-200, 1.0e-201, 1.0e-202, 1.0e-203,
    1.0e-204, 1.0e-205, 1.0e-206, 1.0e-207,
    1.0e-208, 1.0e-209, 1.0e-210, 1.0e-211,
    1.0e-212, 1.0e-213, 1.0e-214, 1.0e-215,
    1.0e-216, 1.0e-217, 1.0e-218, 1.0e-219,
    1.0e-220, 1.0e-221, 1.0e-222, 1.0e-223,
    1.0e-224, 1.0e-225, 1.0e-226, 1.0e-227,
    1.0e-228, 1.0e-229, 1.0e-230, 1.0e-231,
    1.0e-232, 1.0e-233, 1.0e-234, 1.0e-235,
    1.0e-236, 1.0e-237, 1.0e-238, 1.0e-239,
    1.0e-240, 1.0e-241, 1.0e-242, 1.0e-243,
    1.0e-244, 1.0e-245, 1.0e-246, 1.0e-247,
    1.0e-248, 1.0e-249, 1.0e-250, 1.0e-251,
    1.0e-252, 1.0e-253, 1.0e-254, 1.0e-255,
    1.0e-256, 1.0e-257, 1.0e-258, 1.0e-259,
    1.0e-260, 1.0e-261, 1.0e-262, 1.0e-263,
    1.0e-264, 1.0e-265, 1.0e-266, 1.0e-267,
    1.0e-268, 1.0e-269, 1.0e-270, 1.0e-271,
    1.0e-272, 1.0e-273, 1.0e-274, 1.0e-275,
    1.0e-276, 1.0e-277, 1.0e-278, 1.0e-279,
    1.0e-280, 1.0e-281, 1.0e-282, 1.0e-283,
    1.0e-284, 1.0e-285, 1.0e-286, 1.0e-287,
    1.0e-288, 1.0e-289, 1.0e-290, 1.0e-291,
    1.0e-292, 1.0e-293, 1.0e-294, 1.0e-295,
    1.0e-296, 1.0e-297, 1.0e-298, 1.0e-299,
    1.0e-300, 1.0e-301, 1.0e-302, 1.0e-303,
    1.0e-304, 1.0e-305, 1.0e-306, 1.0e-307,
    1.0e-308, 1.0e-309, 1.0e-310, 1.0e-311,
    1.0e-312, 1.0e-313, 1.0e-314, 1.0e-315,
    1.0e-316, 1.0e-317, 1.0e-318, 1.0e-319,
    1.0e-320, 1.0e-321, 1.0e-322, 1.0e-323,
};

constexpr size_t decimals_zero_d = 308;

inline
double assemble_float(int64_t int_p, int64_t frac_p, int64_t exp, size_t frac_len, uint8_t sign) noexcept
{
    double decimal_f = decimals_d[decimals_zero_d + frac_len];
    double decimal_e = decimals_d[decimals_zero_d - exp];
    double sign64 = decimal_e * (1 - sign * 2);
    double res = int_p + decimal_f * frac_p;

    res *= sign64;

    return res;
}

inline
bool str_to_int64(int64_t& res, const char* __restrict str, size_t start, size_t len) noexcept
{
    simd_m128i mask0, mask1;

    simd_m128i step0 = simd_mm_set1_epi8('0');
    simd_m128i conv = simd_mm_loadu_si128((simd_m128i*)str); // this will also cause prefetch

    conv = simd_mm_subs_epu8(conv, step0); // convert digits to bytes

    switch (len) {
    default:
    {
        simd_m128i mask, tmp;

        simd_m128i step1 = simd_mm_set_epi8(10, 1, 10, 1, 10, 1, 10, 1,
                                            10, 1, 10, 1, 10, 1, 10, 1); // assemble tens and ones

        mask0 = simd_mm_setr_epi8( -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,
                                   -9, -10, -11, -12, -13, -14, -15, -16); // shuffle mask

        mask1 = simd_mm_set1_epi8(len);
        mask  = simd_mm_add_epi8(mask0, mask1);   // add len - the shuffle positions will surface

        conv = simd_mm_shuffle_epi8(conv, mask);  // now conv contains a fully prepared value


        conv = simd_mm_maddubs_epi16(conv, step1);

        simd_m128i step2 = simd_mm_set_epi16(100, 1, 100, 1,
                                             100, 1, 100, 1); // assemble tens/ones and hundreds/thousands
        conv = simd_mm_madd_epi16(conv, step2);

        conv = simd_mm_shufflelo_epi16(conv, (3 << 6) | (1 << 4) | (2 << 2) | (0 << 0));
        conv = simd_mm_shufflehi_epi16(conv, (3 << 6) | (1 << 4) | (2 << 2) | (0 << 0)); // Move 16-bit values into adjacent lanes

        simd_m128i step3 = simd_mm_set_epi16(0, 0, 10000, 1, 0, 0, 10000, 1); // assemble tens/ones and tens of thousands
        conv = simd_mm_madd_epi16(conv, step3);

        simd_m128i step4 = simd_mm_set_epi32(0, 100000000, 0, 1);
        conv = simd_mm_mul_epu32(conv, step4);

        tmp = simd_mm_bsrli_si128(conv, 8);
        conv = simd_mm_add_epi64(conv, tmp);
        res = simd_mm_extract_epi64(conv, 0);

        return true;
    }
    case 0:
    {
        res = 0;
        return true;
    }
    case 1:
    {
        res = simd_mm_extract_epi8(conv, 0);
        return true;
    }
    case 2:
    {
        uint32_t mask32 = (1 << 8) | (10 << 0);
        mask0 = simd_mm_loadu_si32(&mask32);
        conv = simd_mm_maddubs_epi16(mask0, conv);
        res = simd_mm_extract_epi16(conv, 0);
        return true;
    }
    case 3:
    {
        uint32_t mask32_0 = (1 << 16) | (1 << 8) | (10 << 0);
        uint32_t mask32_1 = (1 << 16) | (10 << 0);

        mask0 = simd_mm_loadu_si32(&mask32_0);
        mask1 = simd_mm_loadu_si32(&mask32_1);
        conv = simd_mm_maddubs_epi16(mask0, conv);
        conv = simd_mm_madd_epi16(mask1, conv);

        res = simd_mm_extract_epi16(conv, 0);
        return true;
    }
    case 4:
    {
        uint32_t mask32_0 = (1 << 24) | (10 << 16) | (1 << 8) | (10 << 0);
        uint32_t mask32_1 = (1 << 16) | (100 << 0);

        mask0 = simd_mm_loadu_si32(&mask32_0);
        mask1 = simd_mm_loadu_si32(&mask32_1);
        conv = simd_mm_maddubs_epi16(mask0, conv);
        conv = simd_mm_madd_epi16(mask1, conv);

        res = simd_mm_extract_epi16(conv, 0);
        return true;
    }
    }

    return true;
}

//--------------------------------------------------------------------
bool parse_number(double& res, const char * __restrict str_in, size_t len) noexcept
{
    const char * __restrict str_c = str_in;
    const size_t str_len = len;
    bool ret = false;

    uint32_t bitmask = str_get_bitmask(str_c);

    uint8_t char_c;
    size_t int_s = 0;
    size_t int_e = 0;
    size_t frac_s = 0;
    size_t frac_e = 0;
    size_t exp_s = 0;
    size_t exp_e = 0;
    size_t sign_s = 0;
    size_t sign_e = 0;
    uint8_t sign_m;
    int64_t int_i = 0, frac_i = 0, exp_i = 0;
    size_t int_len = 0, frac_len = 0, exp_len = 0;

    char_c = str_extract_byte(str_c, 0);
    sign_m = (char_c == '-') ? 1 : 0;

    if (sign_m != 0) {
        bitmask = set_mask_zero(bitmask, int_s);
        int_s++;
        char_c = str_extract_byte(str_c, int_s);
        if (!char_c
             || !(bit_is_zero(bitmask, int_s) || is_dot(char_c))) {
            // Only allow integer or '.' after '-'
            goto exit_false;
        }
    }

    int_e = pos_non_zero(bitmask);

    int_len = int_e - int_s;

    str_to_int64(int_i, str_c + int_s, int_s, int_len);

    char_c = str_extract_byte(str_c, int_e);

    if (!char_c) {
        goto exit_true;
    }

    if (!(is_exp(char_c) || is_dot(char_c))) {
        goto exit_false;
    }

    frac_s = int_e;
    frac_e = frac_s;

    if (is_dot(char_c)) {
        bitmask = set_mask_zero(bitmask, frac_s);
        frac_s++;

        frac_e = pos_non_zero(bitmask);
        frac_len = frac_e - frac_s;

        str_to_int64(frac_i, str_c + frac_s, frac_s, frac_len);
    }

    char_c = str_extract_byte(str_c, frac_e);

    if (!char_c) {
        goto exit_true;
    }

    if (!is_exp(char_c))
        goto exit_false;

    exp_s = frac_e;
    exp_e = exp_s;

    if (is_exp(char_c)) {
        bitmask = set_mask_zero(bitmask, exp_s);
        exp_s++;

        char_c = str_extract_byte(str_c, exp_s);

        if (is_sign(char_c)) {
            sign_e = (char_c == '-') ? 1 : 0;
            bitmask = set_mask_zero(bitmask, exp_s);
            exp_s++;
        }

        exp_e = pos_non_zero(bitmask);

        exp_len = exp_e - exp_s;

        str_to_int64(exp_i, str_c + exp_s, exp_s, exp_len);

        exp_i = (sign_e) ? -exp_i : exp_i;
    }

    char_c = str_extract_byte(str_c, exp_e);

    if (char_c) {
        goto exit_false;
    }

exit_true:
    res = assemble_float(int_i, frac_i, exp_i, frac_len, sign_m);
    return true;

exit_false:
    return false;
}

bool  parse_number(double& res, const std::string& str) noexcept
{
    return parse_number(res, str.c_str(), str.size());
}

//--------------------------------------------------------------------

std::vector<std::string> generate_random_numbers(size_t& volume, size_t number) {
    std::vector<std::string> lines;
    std::random_device rand_dev;
    std::mt19937 generator;
    std::uniform_real_distribution<double> distr;
    std::ostringstream os;

    os.precision(std::numeric_limits<double>::max_digits10 - 1);
    os.setf(std::ios::scientific);

    volume = 0;
    lines.reserve(number);
    for (size_t i = 0; i < number; i++) {
        os << distr(generator);

        volume += os.str().size();
        lines.push_back(os.str());
        os.str("");
    }

    return lines;
}

typedef char vsi[32];

std::pair<double, double> laptime(const std::vector<std::string>& lines, size_t loops) {
    std::chrono::high_resolution_clock::time_point t_s, t_e;
    double avg_t = 0;
    double min_t = std::numeric_limits<double>::max();

    vsi* strings = new vsi[lines.size()];

    for (size_t i = 0; i < lines.size(); i++) {
        strncpy(strings[i], lines[i].c_str(), sizeof(vsi));
    }


    for (size_t i = 0; i < loops; i++) {
        double res;
        t_s = std::chrono::high_resolution_clock::now();
#if 0
        for (const std::string& line: lines) {
            bool ret = parse_number(res, line);
            if (!ret) {
                std::cout << "!!!BUG!!!" << std::endl;
            }
        }
#else
        for (size_t i = 0; i < lines.size(); i++) {
            bool ret = parse_number(res, strings[i], sizeof(vsi));
            if (!ret) {
                std::cout << "!!!BUG!!!" << std::endl;
            }
        }
#endif
        t_e = std::chrono::high_resolution_clock::now();
        double time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_e - t_s).count();
        avg_t += time;
        min_t = std::min(time, min_t);
    }
    avg_t /= loops;

    return std::make_pair(min_t, avg_t);
}

void pretty_print(double volume, size_t number_of_floats, std::string name, std::pair<double, double> result) {
    double mbytes = volume / (1024. * 1024.);
    std::cout << mbytes * 1000000000 / result.first << " MB/s "
        << (double)number_of_floats * 1000 / result.first << " Mfloat/s "
        << (result.second - result.first) * 100. /result.second << "% "
        << double(result.first) / number_of_floats << " ns/float"
        << std::endl;
}

void speed_test(size_t number_of_floats, size_t loops)
{
    size_t volume;

    std::vector<std::string> lines = generate_random_numbers(volume, number_of_floats);
    std::pair<double, double> result = laptime(lines, loops);
    pretty_print((double)volume, number_of_floats, "turbofloat", result);
}

constexpr double epsilon_d = 2e-16;

inline bool in_ranged(double x, double y)
{
    const double diff = fabs(x - y);
    const double eps = epsilon_d * fmin(fabs(x), fabs(y));

    return diff <= eps;
}

bool turbofloat_test(void)
{
    bool ret = false;

    try {
        double res_d;

        parse_number(res_d, "9897969594939291");
        if (!in_ranged(res_d, 9897969594939291.0)) throw;
        parse_number(res_d, "-1");
        if (!in_ranged(res_d, -1.0)) throw;
        parse_number(res_d, "1");
        if (!in_ranged(res_d, 1.0)) throw;
        parse_number(res_d, "12");
        if (!in_ranged(res_d, 12.0)) throw;
        parse_number(res_d, "123");
        if (!in_ranged(res_d, 123.0)) throw;
        parse_number(res_d, "1234");
        if (!in_ranged(res_d, 1234.0)) throw;
        parse_number(res_d, "12345");
        if (!in_ranged(res_d, 12345.0)) throw;
        parse_number(res_d, "123456");
        if (!in_ranged(res_d, 123456.0)) throw;
        parse_number(res_d, "1234567");
        if (!in_ranged(res_d, 1234567.0)) throw;
        parse_number(res_d, "12345678");
        if (!in_ranged(res_d, 12345678.0)) throw;
        parse_number(res_d, "123456789");
        if (!in_ranged(res_d, 123456789.0)) throw;
        parse_number(res_d, "1234567890");
        if (!in_ranged(res_d, 1234567890.0)) throw;
        parse_number(res_d, "12345678901");
        if (!in_ranged(res_d, 12345678901.0)) throw;
        parse_number(res_d, "123456789012");
        if (!in_ranged(res_d, 123456789012.0)) throw;
        parse_number(res_d, "1234567890123");
        if (!in_ranged(res_d, 1234567890123.0)) throw;
        parse_number(res_d, "12345678901234");
        if (!in_ranged(res_d, 12345678901234.0)) throw;
        parse_number(res_d, "123456789012345");
        if (!in_ranged(res_d, 123456789012345.0)) throw;
        parse_number(res_d, "1234567890123456");
        if (!in_ranged(res_d, 1234567890123456.0)) throw;
        parse_number(res_d, "-97969594939291");
        if (!in_ranged(res_d, -97969594939291.0)) throw;

        parse_number(res_d, "9999999999999999");
        if (!in_ranged(res_d, 9999999999999999.0)) throw;

        parse_number(res_d, "234689");
        if (!in_ranged(res_d, 234689.0)) throw;
        parse_number(res_d, "234567801627364");
        if (!in_ranged(res_d, 234567801627364.0)) throw;

        parse_number(res_d, "23456781");
        if (!in_ranged(res_d, 23456781.0)) throw;
        parse_number(res_d, "2345678175");
        if (!in_ranged(res_d, 2345678175.0)) throw;

        parse_number(res_d, "1.0");
        if (!in_ranged(res_d, 1.0)) throw;
        parse_number(res_d, "-12345689");
        if (!in_ranged(res_d, -12345689.0)) throw;
        parse_number(res_d, "-12345689.");
        if (!in_ranged(res_d, -12345689.0)) throw;
        parse_number(res_d, "12345689");
        if (!in_ranged(res_d, 12345689.0)) throw;
        parse_number(res_d, ".12345689");
        if (!in_ranged(res_d, .12345689)) throw;
        parse_number(res_d, "-.12345689");
        if (!in_ranged(res_d, -.12345689)) throw;
        parse_number(res_d, "-0.12345689");
        if (!in_ranged(res_d, -.12345689)) throw;
        parse_number(res_d, "-123456.789");
        if (!in_ranged(res_d, -123456.789)) throw;
        parse_number(res_d, "3456321.998");
        if (!in_ranged(res_d, 3456321.998)) throw;
        parse_number(res_d, "-2.434e252");
        if (!in_ranged(res_d, -2.434e252)) throw;
        parse_number(res_d, "-2.34232e-305");
        if (!in_ranged(res_d, -2.34232e-305)) throw;
        parse_number(res_d, "-65.3232e+184");
        if (!in_ranged(res_d, -65.3232e+184)) throw;
        parse_number(res_d, "55.3232e+304");
        if (!in_ranged(res_d, 55.3232e+304)) throw;
        parse_number(res_d, "-55.3232e+00");
        if (!in_ranged(res_d, -55.3232)) throw;
        parse_number(res_d, "000e+00");
        if (!in_ranged(res_d, 0.0)) throw;
        parse_number(res_d, "-000e+00");
        if (!in_ranged(res_d, -0.0)) throw;
        parse_number(res_d, "00.00e+00");
        if (!in_ranged(res_d, 0.0)) throw;
        parse_number(res_d, "0.00e-00");
        if (!in_ranged(res_d, 0.0)) throw;

        ret = true;
    }
    catch (...) {
        ret = false;
    }

    return ret;
}


int main()
{
    double res_d;

    std::cout << "Start..." << std::endl;

    turbofloat_test();

    speed_test(10000, 25000);
}
