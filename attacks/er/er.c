#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/ipc.h>
#include <stdint.h>
#include "../cacheutils.h"

size_t start = 0;
size_t keystate = 0;
size_t kpause = 0;
void evictandreload(void* addr)
{
  size_t time = rdtsc_begin();
  maccess(addr);
  size_t delta = rdtsc_end() - time;
  evict(addr);
  if (delta < 100)
  {
    if (kpause > 1000)
    {
      printf("CACHE HIT - Δ = %-4lu cycles | idle = %-5lu loops | t = %6lu µs\n", delta, kpause, ((time - start) / 2600));
 
      keystate = (keystate+1) % 2;
    }
    kpause = 0;
  }
  else
    kpause++;
}

int main(int argc, char** argv)
{
  char* name = argv[1];
  char* offsetp = argv[2];
  if (argc != 3)
    return 1;
  unsigned int offset = 0;
  !sscanf(offsetp,"%x",&offset);
  int fd = open(name,O_RDONLY);
  if (fd < 3)
    return 2;
  unsigned char* addr = (unsigned char*)mmap(0, 64*1024*1024, PROT_READ, MAP_SHARED, fd, 0);
  if (addr == (void*)-1)
    return 3;
  printf("Monitoring %s + 0x%x\n", name, offset);
  start = rdtsc();
  while(1)
  {
    evictandreload(addr + offset);
    sched_yield();
  }
  return 0;
}
