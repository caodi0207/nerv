package = "nerv"
version = "scm-1"
source = {
    url = "https://github.com/Determinant/nerv.git"
}
description = {
    summary = "A Lua-based toolkit dedicated to high-performance deep neural network learning",
    detailed = [[
    ]],
    homepage = "https://github.com/Determinant/nerv",
    license = "BSD"
}
dependencies = {
    "lua >= 5.1"
}
build = {
    type = "make",
    build_variables = {
        CFLAGS="$(CFLAGS)",
        LIBFLAG="$(LIBFLAG)",
        LUA_LIBDIR="$(LUA_LIBDIR)",
        LUA_BINDIR="$(LUA_BINDIR)",
        LUA_INCDIR="$(LUA_INCDIR)",
        LUA="$(LUA)",
    },
    install_variables = {
        LUA_BINDIR="$(LUA_BINDIR)",
        INST_PREFIX="$(PREFIX)",
        INST_BINDIR="$(BINDIR)",
        INST_LIBDIR="$(LIBDIR)",
        INST_LUADIR="$(LUADIR)",
        INST_CONFDIR="$(CONFDIR)",
    },
    install = {
        bin = {"nerv"}
    }
}
