require 'libnerv'

function nerv.error(fmt, ...)
    error("[nerv] internal error: " .. fmt .. "\n", ...)
end

function nerv.error_method_not_implemented()
    nerv.error("method not implemented");
end

function nerv.printf(fmt, ...)
    io.write(string.format(fmt, ...))
end

function nerv.mesg_with_timestamp(fmt, ...)
    nerv.printf(
        string.format("(%s)[nerv] info: %s\n",
            os.date("%H:%M:%S %F"), fmt), ...)
end

function nerv.info(fmt, ...)
    nerv.printf(
        string.format("(%s)[nerv] info: %s\n",
            os.date("%H:%M:%S %F"), fmt), ...)
end

function nerv.warning(fmt, ...)
    nerv.printf(
        string.format("(%s)[nerv] warning: %s\n",
            os.date("%H:%M:%S %F"), fmt), ...)
end

-- Torch C API wrapper
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

function nerv.get_type(tname)
    return assert(loadstring("return " .. tname))()
end

function nerv.is_type(obj, tname)
    local mt0 = nerv.getmetatable(tname)
    local mt = getmetatable(obj)
    while mt do
        if mt == mt0 then
            return true
        end
        mt = getmetatable(mt)
    end
    return false
end

function nerv.dirname(filename)
    if filename:match(".-/.-") then
        local name = string.gsub(filename, "(.*/)(.*)", "%1")
        return name
    else
        return ''
    end
end

function nerv.include(filename)
    local caller = debug.getinfo(2, "S").source:sub(2)
    dofile(nerv.dirname(caller) .. filename)
end

nerv.include('matrix/init.lua')
nerv.include('io/init.lua')
nerv.include('layer/init.lua')
nerv.include('nn/init.lua')
