// SPDX-License-Identifier: Apache-2.0
// Freestanding Linux/AArch64 tests: no cross libc or installed SDK required.
#include <stddef.h>
#include <stdint.h>

extern float lance_dot_f32_sve(const float *, const float *, size_t);
extern float lance_l2_f32_sve(const float *, const float *, size_t);
extern uint32_t lance_dot_u8_sve(const uint8_t *, const uint8_t *, size_t);
extern uint32_t lance_l2_u8_sve(const uint8_t *, const uint8_t *, size_t);
extern void lance_sum_f32_rows_sve(const uint8_t *, const float *, size_t,
                                  size_t, const uint32_t *, size_t, float *);
extern void lance_sum_hacc_sve2(const uint8_t *, const uint8_t *, size_t,
                               size_t, uint32_t *);

static long syscall6(long nr, long a, long b, long c, long d, long e, long f) {
  register long x8 __asm__("x8") = nr;
  register long x0 __asm__("x0") = a;
  register long x1 __asm__("x1") = b;
  register long x2 __asm__("x2") = c;
  register long x3 __asm__("x3") = d;
  register long x4 __asm__("x4") = e;
  register long x5 __asm__("x5") = f;
  __asm__ volatile("svc 0" : "+r"(x0)
                   : "r"(x8), "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5)
                   : "memory", "cc");
  return x0;
}

static void fail(int code) {
  syscall6(93, code, 0, 0, 0, 0, 0);
  __builtin_unreachable();
}

void *memset(void *dst, int c, size_t n) {
  uint8_t *p = dst;
  for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)c;
  return dst;
}

void *memcpy(void *dst, const void *src, size_t n) {
  uint8_t *d = dst;
  const uint8_t *s = src;
  for (size_t i = 0; i < n; ++i) d[i] = s[i];
  return dst;
}

static uint32_t state = 17;
static uint32_t random32(void) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static uint32_t bits(float x) {
  union { float f; uint32_t u; } value = { .f = x };
  return value.u;
}

static double absd(double x) { return x < 0 ? -x : x; }

static void test_distances(void) {
  // Beyond 66051 bytes, the maximum u8 dot/L2 sum overflows u32.
  static uint8_t a[70001], b[70001];
  static float x[4097], y[4097];
  for (size_t i = 0; i < sizeof(a); ++i) {
    a[i] = (uint8_t)random32(); b[i] = (uint8_t)random32();
  }
  for (size_t n = 0; n <= 1025; ++n) {
    uint64_t dot = 0, l2 = 0;
    for (size_t i = 0; i < n; ++i) {
      dot += (uint32_t)a[i] * b[i];
      int d = (int)a[i] - (int)b[i]; l2 += (unsigned)(d * d);
    }
    if (lance_dot_u8_sve(a, b, n) != (uint32_t)dot) fail(11);
    if (lance_l2_u8_sve(a, b, n) != (uint32_t)l2) fail(12);
  }
  memset(a, 255, sizeof(a)); memset(b, 255, sizeof(b));
  if (lance_dot_u8_sve(a, b, sizeof(a)) != (uint32_t)(65025ULL * sizeof(a))) fail(13);
  memset(b, 0, sizeof(b));
  if (lance_l2_u8_sve(a, b, sizeof(a)) != (uint32_t)(65025ULL * sizeof(a))) fail(14);
  for (size_t i = 0; i < 4097; ++i) {
    x[i] = ((int)(random32() % 2001) - 1000) / 173.0f;
    y[i] = ((int)(random32() % 2001) - 1000) / 191.0f;
  }
  for (size_t n = 0; n <= 4097; n += n < 129 ? 1 : 31) {
    double dot = 0, l2 = 0, scale = 0;
    for (size_t i = 0; i < n; ++i) {
      double product = (double)x[i] * y[i]; dot += product; scale += absd(product);
      float d = x[i] - y[i]; l2 += (double)d * d;
    }
    if (absd(lance_dot_f32_sve(x, y, n) - dot) > 2e-6 * scale + 1e-6) fail(15);
    if (absd(lance_l2_f32_sve(x, y, n) - l2) > 2e-6 * l2 + 1e-6) fail(16);
  }
  x[0] = __builtin_nanf(""); y[0] = 1;
  if (!__builtin_isnan(lance_dot_f32_sve(x, y, 1))) fail(17);
  if (!__builtin_isnan(lance_l2_f32_sve(x, y, 1))) fail(18);
  x[0] = __builtin_inff();
  if (lance_l2_f32_sve(x, y, 1) != __builtin_inff()) fail(19);
}

static void test_lookup(void) {
  enum { N = 96, COLS = 257 };
  static uint8_t codes[N * COLS], hacc[64 * COLS];
  static float tables[32 * COLS], out[N + 1];
  static uint32_t rows[N], hout[N + 1], expected[N];
  for (size_t i = 0; i < sizeof(codes); ++i) codes[i] = (uint8_t)random32();
  for (size_t i = 0; i < sizeof(hacc); ++i) hacc[i] = (uint8_t)random32();
  for (size_t i = 0; i < 32 * COLS; ++i)
    tables[i] = ((int)(random32() % 2001) - 1000) / 117.0f;
  // Include repeated and unsorted rows.
  for (size_t i = 0; i < N; ++i) rows[i] = random32() % N;
  const size_t widths[] = {0, 1, 2, 3, 17, 128, 257};
  for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w) {
    size_t cols = widths[w];
    for (size_t count = 0; count <= N; ++count) {
      out[count] = 12345;
      lance_sum_f32_rows_sve(codes, tables, N, cols, rows, count, out);
      if (out[count] != 12345) fail(21);
      for (size_t r = 0; r < count; ++r) {
        float sum = -0.0f;
        for (size_t c = 0; c < cols; ++c) {
          uint8_t code = codes[c * N + rows[r]];
          sum += tables[c * 32 + (code & 15)] + tables[c * 32 + 16 + (code >> 4)];
        }
        if (bits(sum) != bits(out[r])) fail(22);
      }
    }
    for (size_t n = 0; n <= N; n += 32) {
      memset(expected, 0, sizeof(expected));
      for (size_t block = 0; block < n; block += 32) {
        for (size_t c = 0; c < cols; ++c) {
          for (size_t j = 0; j < 16; ++j) {
            uint8_t c0 = codes[block * cols + c * 32 + j];
            uint8_t c1 = codes[block * cols + c * 32 + 16 + j];
            size_t row = block + j / 2 + (j % 2) * 8;
            const uint8_t *t = hacc + c * 64;
            expected[row] += t[c0 & 15] + ((uint32_t)t[32 + (c0 & 15)] << 8)
                + t[16 + (c1 & 15)] + ((uint32_t)t[48 + (c1 & 15)] << 8);
            expected[row + 16] += t[c0 >> 4] + ((uint32_t)t[32 + (c0 >> 4)] << 8)
                + t[16 + (c1 >> 4)] + ((uint32_t)t[48 + (c1 >> 4)] << 8);
          }
        }
      }
      hout[n] = 0xdeadbeef;
      lance_sum_hacc_sve2(codes, hacc, cols, n, hout);
      if (hout[n] != 0xdeadbeef) fail(23);
      for (size_t i = 0; i < n; ++i) if (hout[i] != expected[i]) fail(24);
    }
  }
  // Signed zero must survive a nonempty table, too.
  for (size_t i = 0; i < 32; ++i) tables[i] = -0.0f;
  lance_sum_f32_rows_sve(codes, tables, N, 1, rows, N, out);
  for (size_t i = 0; i < N; ++i) if (bits(out[i]) != 0x80000000) fail(25);
  // Every table entry at 65535 catches byte carry and u16 sum overflow bugs.
  memset(hacc, 255, sizeof(hacc));
  lance_sum_hacc_sve2(codes, hacc, COLS, N, hout);
  for (size_t i = 0; i < N; ++i) if (hout[i] != 2U * COLS * 65535U) fail(26);
}

static uint8_t *guarded(void) {
  long p = syscall6(222, 0, 8192, 3, 0x22, -1, 0); // mmap anonymous RW
  if (p < 0 && p > -4096) fail(30);
  if (syscall6(226, p + 4096, 4096, 0, 0, 0, 0)) fail(31); // PROT_NONE
  return (uint8_t *)p;
}

static void test_guard_pages(void) {
  uint8_t *a = guarded(), *b = guarded(), *c = guarded(), *d = guarded();
  for (size_t n = 0; n < 257; ++n) {
    uint8_t *x = a + 4096 - n, *y = b + 4096 - n;
    memset(x, 255, n); memset(y, 1, n);
    if (lance_dot_u8_sve(x, y, n) != 255U * n) fail(32);
    if (lance_l2_u8_sve(x, y, n) != 64516U * n) fail(33);
    float *xf = (float *)(a + 4096) - n, *yf = (float *)(b + 4096) - n;
    for (size_t i = 0; i < n; ++i) { xf[i] = 2; yf[i] = 1; }
    if (lance_dot_f32_sve(xf, yf, n) != 2.0f * n) fail(34);
    if (lance_l2_f32_sve(xf, yf, n) != (float)n) fail(35);
  }
  for (size_t n = 1; n < 65; ++n) {
    uint8_t *codes = a + 4096 - n;
    float *table = (float *)(b + 4096) - 32;
    uint32_t *rows = (uint32_t *)(c + 4096) - n;
    float *out = (float *)(d + 4096) - n;
    memset(codes, 255, n);
    for (size_t i = 0; i < 32; ++i) table[i] = (float)i;
    for (size_t i = 0; i < n; ++i) rows[i] = (uint32_t)(n - i - 1);
    lance_sum_f32_rows_sve(codes, table, n, 1, rows, n, out);
    for (size_t i = 0; i < n; ++i) if (out[i] != 46) fail(36);
  }
  memset(a + 4096 - 32, 255, 32); memset(b + 4096 - 64, 255, 64);
  uint32_t *out = (uint32_t *)(d + 4096) - 32;
  lance_sum_hacc_sve2(a + 4096 - 32, b + 4096 - 64, 1, 32, out);
  for (size_t i = 0; i < 32; ++i) if (out[i] != 131070) fail(37);
  syscall6(215, (long)a, 8192, 0, 0, 0, 0);
  syscall6(215, (long)b, 8192, 0, 0, 0, 0);
  syscall6(215, (long)c, 8192, 0, 0, 0, 0);
  syscall6(215, (long)d, 8192, 0, 0, 0, 0);
}

void _start(void) {
  // Changing VL between calls catches cached-width assumptions. This test
  // driver is baseline ARMv8 and holds no live SVE values across prctl.
  const unsigned widths[] = {16, 32, 64, 128, 256};
  for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); ++i) {
    long result = syscall6(167, 50, widths[i], 0, 0, 0, 0); // prctl(PR_SVE_SET_VL)
    if (result < 0 || (result & 0xffff) != widths[i]) fail(40);
    test_distances(); test_lookup(); test_guard_pages();
  }
  static const char message[] = "PASS: six kernels, VL 128/256/512/1024/2048, tails, overflow, lookup order, guard pages\n";
  syscall6(64, 1, (long)message, sizeof(message) - 1, 0, 0, 0);
  fail(0);
}
