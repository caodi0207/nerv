require 'libnerv'
require 'matrix.init'
nerv.class = require 'pl.class'
nerv.utils = require 'pl.utils'

function nerv.error(fmt, ...)
    error(nerv.utils.printf("Nerv internal error: " .. fmt, ...))
end

function nerv.error_method_not_implement()
    nerv.error("method not implemented");
end
