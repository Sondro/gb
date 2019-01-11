#include <stdint.h>
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef  int8_t s8; typedef  int16_t s16; typedef  int32_t s32; typedef  int64_t s64;

//registers
typedef struct {
  // regs
  union {
    struct {
      u8 C; u8 B;
      u8 E; u8 D;
      u8 L; u8 H;
      union {
        struct { u8 unused:4; u8 FC:1; u8 FH:1; u8 FN:1; u8 FZ:1;};
        u8 F; }; u8 A;
      u16 SP; u16 PC;
    };
    u16 regs[6];
  };
  // extra registers for handling register transfers
  u8 src_reg;
  u8 dst_reg;
  u8 unimpl;

  u8* rom;
  u8 vram[0x2000];
  u8 hram[256];

  u32 cpu_ticks;

} gb;

#define BC (g->regs[0])
#define DE (g->regs[1])
#define HL (g->regs[2])
#define AF (g->regs[3])
#define SP (g->regs[4])
#define PC (g->regs[5])

#define B (g->B)
#define C (g->C)
#define D (g->D)
#define E (g->E)
#define H (g->H)
#define L (g->L)
#define A (g->A)
#define F (g->F)

#define fC (g->FC)
#define fN (g->FN)
#define fZ (g->FZ)
#define fH (g->FH)
