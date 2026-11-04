-- tables.lua - pandoc filter for build_tacl.py: two-column floats for tables and figures.
--
-- A pipe table becomes a float holding a tabular of ragged p-columns (booktabs rules, \small),
-- spanning both columns (table*) when its natural width exceeds one column. A paragraph that
-- follows a table and opens with "**Table N.**", or follows an image and opens with
-- "**Figure N.**", becomes the float's caption, kept as written (\caption* with the draft's own
-- numbering, so "Table B1" stays "Table B1"). A figure (pandoc 3 reads a lone image as one) is
-- set at column width; blocks are filtered bottom-up, so only the Figure itself is matched.

local COLUMN_CHARS = 50     -- characters of \small text across one TACL column (7.7 cm)
local PAGE_CHARS = 104      -- across both columns (16 cm)
local TABCOLSEP = '3.5pt'   -- half the space between columns (LaTeX's default is 6pt)
local PAD = 1.6             -- 2\tabcolsep, in characters
local MAX_LINES = 46        -- the estimated lines of a table set at \small on one page
local SHORT = 22            -- a column whose cells are all this short is never wrapped
                            -- (an estimate with its interval, '+0.03 [-1.80, +1.86]')

local function latex(blocks)
  local out = pandoc.write(pandoc.Pandoc(blocks), 'latex')
  out = out:gsub('^%s+', ''):gsub('%s+$', '')
  return (out:gsub('\n\n+', ' \\newline '))
end

local function cell_text(cell)
  return latex(cell.contents)
end

local function length(s)
  return utf8.len(s) or #s
end

local function longest_word(s)
  local m = 0
  for w in s:gmatch('%S+') do m = math.max(m, length(w)) end
  return m
end

local function rows_of(tbl)
  local rows = {}
  for _, row in ipairs(tbl.head.rows) do table.insert(rows, {row = row, head = true}) end
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.body) do table.insert(rows, {row = row, head = false}) end
  end
  return rows
end

local function caption_of(block, word)
  -- the caption paragraph's inlines when `block` is a Para opening with Strong "<word> ..."
  if block == nil or block.t ~= 'Para' then return nil end
  local first = block.content[1]
  if first == nil or first.t ~= 'Strong' then return nil end
  local label = pandoc.utils.stringify(first)
  if label:match('^' .. word .. ' [%w%.]+%.$') then return block.content end
  return nil
end

-- column widths in characters: from the floors, the remaining room handed out one character
-- at a time to the column that most lowers the table's estimated height (each row's tallest
-- cell, in lines); when no character lowers it, the rest is shared by the text each column
-- holds. Returns the widths and the estimated height.
local function allocate(lens, floor, room)
  local ncol = #floor
  local w, used = {}, 0
  for i = 1, ncol do w[i] = floor[i]; used = used + w[i] end
  local function height()
    local h = 0
    for _, row in ipairs(lens) do
      local m = 1
      for i, len in ipairs(row) do m = math.max(m, math.ceil(len / w[i])) end
      h = h + m
    end
    return h
  end
  while used < room do
    local best, best_h = nil, height()
    for i = 1, ncol do
      w[i] = w[i] + 1
      local h = height()
      w[i] = w[i] - 1
      if h < best_h then best, best_h = i, h end
    end
    if best == nil then
      local text, total = {}, 0
      for i = 1, ncol do
        text[i] = 0
        for _, row in ipairs(lens) do text[i] = text[i] + row[i] end
        total = total + text[i]
      end
      local rest = room - used
      for i = 1, ncol do w[i] = w[i] + rest * text[i] / total end
      used = room
    else
      w[best] = w[best] + 1
      used = used + 1
    end
  end
  return w, height()
end

local function table_float(tbl, caption)
  local rows = rows_of(tbl)
  local ncol = #tbl.colspecs
  -- each cell's length and each column's longest word, in characters
  local lens, body_max, floor = {}, {}, {}
  for i = 1, ncol do body_max[i], floor[i] = 1, 1 end
  for _, r in ipairs(rows) do
    local row = {}
    for i, cell in ipairs(r.row.cells) do
      local text = pandoc.utils.stringify(cell.contents)
      row[i] = length(text)
      floor[i] = math.max(floor[i], longest_word(text) + 1)
      if not r.head then body_max[i] = math.max(body_max[i], row[i]) end
    end
    table.insert(lens, row)
  end
  -- a short column (every body cell at most SHORT characters) is never wrapped
  for i = 1, ncol do
    if body_max[i] <= SHORT then floor[i] = math.max(floor[i], body_max[i] + 1) end
  end
  local natural = 0
  for i = 1, ncol do natural = natural + body_max[i] + 1 + PAD end
  local wide = natural > COLUMN_CHARS
  local chars = wide and PAGE_CHARS or COLUMN_CHARS
  -- at \small; a table that would not fit a page so is set at \footnotesize (9 pt, about
  -- 10/9 the characters per line; footnotes' size) with its rows 10% closer
  local size, scale = '\\small', 1
  local w, h = allocate(lens, floor, chars - PAD * ncol)
  if h > MAX_LINES then
    size, scale = '\\footnotesize\\renewcommand{\\arraystretch}{0.9}', 10 / 9
    w, h = allocate(lens, floor, (chars - PAD * ncol) * scale)
  end
  local sum = 0
  for i = 1, ncol do sum = sum + w[i] + PAD * scale end
  local spec = {}
  for i = 1, ncol do
    table.insert(spec, string.format(
      '>{\\raggedright\\arraybackslash}p{\\dimexpr%.3f\\linewidth-2\\tabcolsep\\relax}',
      (w[i] + PAD * scale) / sum))
  end
  local lines = {}
  local env = wide and 'table*' or 'table'
  table.insert(lines, '\\begin{' .. env .. '}[tp]')
  table.insert(lines, '\\centering' .. size .. '\\setlength{\\tabcolsep}{' .. TABCOLSEP .. '}')
  table.insert(lines, '\\begin{tabular}{' .. table.concat(spec) .. '}')
  table.insert(lines, '\\toprule')
  local in_head = true
  for _, r in ipairs(rows) do
    if in_head and not r.head then
      table.insert(lines, '\\midrule')
      in_head = false
    end
    local cells = {}
    for _, cell in ipairs(r.row.cells) do
      local text = cell_text(cell)
      if r.head then text = '\\textbf{' .. text .. '}' end
      table.insert(cells, text)
    end
    table.insert(lines, table.concat(cells, ' & ') .. ' \\\\')
  end
  table.insert(lines, '\\bottomrule')
  table.insert(lines, '\\end{tabular}')
  if caption then
    table.insert(lines, '\\caption*{' .. latex({pandoc.Plain(caption)}) .. '}')
  end
  table.insert(lines, '\\end{' .. env .. '}')
  return pandoc.RawBlock('latex', table.concat(lines, '\n'))
end

local function image_of(block)
  if block == nil or (block.t ~= 'Para' and block.t ~= 'Plain') then return nil end
  if #block.content == 1 and block.content[1].t == 'Image' then return block.content[1] end
  return nil
end

local function figure_float(image, caption)
  local lines = {'\\begin{figure}[tp]', '\\centering',
                 '\\includegraphics[width=\\columnwidth]{' .. image.src .. '}'}
  if caption then
    table.insert(lines, '\\caption*{' .. latex({pandoc.Plain(caption)}) .. '}')
  end
  table.insert(lines, '\\end{figure}')
  return pandoc.RawBlock('latex', table.concat(lines, '\n'))
end

-- a code block wraps its long lines within the column (fvextra), at \\small
function CodeBlock(block)
  return pandoc.RawBlock('latex', '\\begin{Verbatim}[breaklines,breakanywhere,fontsize=\\small]\n'
                         .. block.text .. '\n\\end{Verbatim}')
end

function Blocks(blocks)
  local out, i = {}, 1
  while i <= #blocks do
    local b = blocks[i]
    if b.t == 'Table' then
      local caption = caption_of(blocks[i + 1], 'Table')
      table.insert(out, table_float(b, caption))
      i = i + (caption and 2 or 1)
    elseif b.t == 'Figure' then
      -- pandoc 3 reads a lone image with alt text as a Figure
      local image
      for _, inner in ipairs(b.content) do
        image = image or image_of(inner)
      end
      if image == nil then error('a figure without an image') end
      local caption = caption_of(blocks[i + 1], 'Figure')
      table.insert(out, figure_float(image, caption))
      i = i + (caption and 2 or 1)
    else
      table.insert(out, b)
      i = i + 1
    end
  end
  return out
end
