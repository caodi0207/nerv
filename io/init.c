#include "../common.h"

extern void nerv_param_file_init(lua_State *L);
void nerv_param_init(lua_State *L) {
    nerv_param_file_init(L);
}
