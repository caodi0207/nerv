require 'libnerv'
require 'matrix.init'
require 'io.init'
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

function table.val_to_str(v)
  if "string" == type(v) then
    v = string.gsub(v, "\n", "\\n")
    if string.match(string.gsub(v,"[^'\"]",""), '^"+$') then
      return "'" .. v .. "'"
    end
    return '"' .. string.gsub(v,'"', '\\"') .. '"'
  else
    return "table" == type(v) and table.tostring(v) or
      tostring(v)
  end
end

function table.key_to_str (k)
  if "string" == type(k) and string.match(k, "^[_%a][_%a%d]*$") then
    return k
  else
    return "[" .. table.val_to_str(k) .. "]"
  end
end

function table.tostring(tbl)
  local result, done = {}, {}
  for k, v in ipairs(tbl) do
    table.insert(result, table.val_to_str(v))
    done[k] = true
  end
  for k, v in pairs(tbl) do
    if not done[k] then
      table.insert(result,
        table.key_to_str(k) .. "=" .. table.val_to_str(v))
    end
  end
  return "{" .. table.concat(result, ",") .. "}"
end
