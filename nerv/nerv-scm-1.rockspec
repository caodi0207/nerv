package = "nerv"
version = "scm-1"
source = {
    url = "..." -- We don't have one yet
}
description = {
    summary = "An example for the LuaRocks tutorial.",
    detailed = [[
    ]],
    homepage = "https://github.com/Determinant/nerv", -- We don't have one yet
    license = "BSD" -- or whatever you like
}
dependencies = {
    "lua >= 5.1"
    -- If you depend on other rocks, add them here
}
build = {
    -- We'll start here.
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
