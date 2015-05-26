local LinearTransParam = nerv.class('nerv.LinearTransParam', 'nerv.Param')
local BiasParam = nerv.class('nerv.BiasParam', 'nerv.LinearTransParam')
local AffineLayer = nerv.class('nerv.AffineLayer', 'nerv.Layer')

function LinearTransParam:read(pcdata)
    self.trans = nerv.CuMatrixFloat.new_from_host(nerv.MMatrixFloat.load(pcdata))
end

function LinearTransParam:write(pfhandle)
    self.trans:new_to_host():save(pfhandle)
end
