#include "../lib/common.h"

extern void nerv_chunk_file_init(lua_State *L);
void nerv_io_init(lua_State *L) {
    nerv_chunk_file_init(L);
}
