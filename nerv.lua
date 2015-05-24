require 'libnerv'
require 'matrix.init'
-- nerv.class = require 'pl.class'
nerv.utils = require 'pl.utils'

function nerv.error(fmt, ...)
    error(nerv.utils.printf("Nerv internal error: " .. fmt .. "\n", ...))
end

function nerv.error_method_not_implement()
    nerv.error("method not implemented");
end

function nerv.class(tname, parenttname)

   local function constructor(...)
      local self = {}
      nerv.setmetatable(self, tname)
      if self.__init then
         self:__init(...)
      end
      return self
   end

   local function factory()
      local self = {}
      nerv.setmetatable(self, tname)
      return self
   end

   local mt = nerv.newmetatable(tname, parenttname, constructor, nil, factory)
   local mpt
   if parenttname then
      mpt = nerv.getmetatable(parenttname)
   end
   return mt, mpt
end
