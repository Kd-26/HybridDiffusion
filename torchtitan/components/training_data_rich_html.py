# Copyright (c) HybridDiffusion contributors.
#
# Licensed under the repository License; see LICENSE in the repository root.

import html
import json
from typing import Any

import torch


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; }
h1, h2, h3, h4 { margin: 0.8rem 0 0.5rem; }
.muted { color: #6b7280; }
.card { border: 1px solid #d1d5db; border-radius: 10px; padding: 16px; margin: 16px 0; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.compare-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; align-items: start; }
.compare-panel { display: grid; gap: 10px; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef2ff; color: #3730a3; margin-right: 6px; margin-bottom: 6px; }
pre { white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 12px; border-radius: 8px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin-top: 10px; }
th, td { border: 1px solid #d1d5db; padding: 6px 8px; vertical-align: top; font-size: 12px; }
th { background: #f3f4f6; }
.tok-piece { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.section-note { margin: 8px 0 0; color: #374151; }
.scroll-pane { max-height: 420px; overflow: auto; border: 1px solid #d1d5db; border-radius: 8px; padding: 10px; background: #ffffff; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 8px 0 12px; }
.legend-item { display: inline-flex; align-items: center; gap: 8px; font-size: 12px; color: #374151; }
.legend-swatch { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #d1d5db; box-sizing: border-box; }
.special-summary { display: grid; gap: 4px; margin: 6px 0 12px; font-size: 12px; color: #374151; }
.canvas-widget { display: grid; gap: 10px; }
.canvas-host { width: 100%; }
.canvas-popup { position: fixed; display: none; z-index: 1000; white-space: pre-wrap; word-break: break-word; background: rgba(255, 255, 255, 0.98); color: #111827; border: 1px solid #cbd5e1; box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18); border-radius: 10px; padding: 12px 14px; font-size: 12px; line-height: 1.45; max-width: min(420px, calc(100vw - 24px)); max-height: min(60vh, 520px); overflow: auto; }
.canvas-popup.visible { display: block; }
@media (max-width: 1200px) { .grid, .compare-grid { grid-template-columns: 1fr; } }
"""


def _to_inline_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _get_token_piece(tokenizer: Any, token_id: int) -> str:
    direct = getattr(tokenizer, "id_to_token", None)
    if callable(direct):
        piece = direct(token_id)
        if piece is not None:
            return piece

    inner = getattr(tokenizer, "tokenizer", None)
    fn = getattr(inner, "id_to_token", None)
    if callable(fn):
        piece = fn(token_id)
        return piece if piece is not None else ""
    return ""


def _get_token_display(tokenizer: Any, token_id: int) -> str:
    try:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        if text:
            return text.replace("\n", "\\n")
    except TypeError:
        try:
            text = tokenizer.decode([int(token_id)])
            if text:
                return text.replace("\n", "\\n")
        except Exception:
            pass
    except Exception:
        pass
    piece = _get_token_piece(tokenizer, token_id)
    return piece if piece else f"<{token_id}>"


def _get_special_token_kind(
    tokenizer: Any,
    token_id: int,
    *,
    mask_token_id: int,
    pad_token_id: int | None,
) -> str | None:
    bos_id = getattr(tokenizer, "bos_id", None)
    eos_id = getattr(tokenizer, "eos_id", None)
    token_piece = _get_token_piece(tokenizer, int(token_id))
    decoded_token = _get_token_display(tokenizer, int(token_id))
    if bos_id is not None and token_id == bos_id:
        return "bos"
    if eos_id is not None and token_id == eos_id:
        return "eos"
    if pad_token_id is not None and token_id == pad_token_id:
        return "pad"
    if token_id == mask_token_id:
        return "mask"
    if decoded_token == "\\n" or token_piece in {"Ċ", "<0x0A>"}:
        return "newline"
    if token_piece.startswith("<|") and token_piece.endswith("|>"):
        return "chat"
    return None


def _special_fill(kind: str) -> str:
    return {
        "bos": "#8b5cf6",
        "eos": "#f59e0b",
        "pad": "#4b5563",
        "mask": "#d946ef",
        "newline": "#14b8a6",
        "chat": "#06b6d4",
    }[kind]


def _special_semantic_role(kind: str, decoded_token: str) -> str:
    if kind == "pad":
        return f"PAD (training padding token; raw tokenizer decode: {decoded_token})"
    if kind == "eos":
        return f"IM_END / EOS (raw tokenizer decode: {decoded_token})"
    if kind == "bos":
        return f"BOS (raw tokenizer decode: {decoded_token})"
    if kind == "mask":
        return f"MASK token id (raw tokenizer decode: {decoded_token})"
    if kind == "newline":
        return f"Newline token (raw tokenizer decode: {decoded_token})"
    if kind == "chat":
        return f"Chat-template special token (raw tokenizer decode: {decoded_token})"
    return "ordinary"


def _count_special_tokens(
    tokenizer: Any,
    token_ids: list[int],
    *,
    mask_token_id: int,
    pad_token_id: int | None,
) -> dict[str, int]:
    counts = {"bos": 0, "eos": 0, "pad": 0, "mask": 0, "newline": 0, "chat": 0}
    for token_id in token_ids:
        kind = _get_special_token_kind(
            tokenizer,
            int(token_id),
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )
        if kind in counts:
            counts[kind] += 1
    return counts


def _collect_special_token_positions(
    tokenizer: Any,
    token_ids: list[int],
    *,
    mask_token_id: int,
    pad_token_id: int | None,
) -> dict[str, list[int]]:
    positions = {"bos": [], "eos": [], "pad": [], "mask": [], "newline": [], "chat": []}
    for idx, token_id in enumerate(token_ids):
        kind = _get_special_token_kind(
            tokenizer,
            int(token_id),
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )
        if kind in positions:
            positions[kind].append(idx)
    return positions


def _render_special_position_summary(positions: dict[str, list[int]], *, limit: int = 12) -> str:
    labels = [
        ("bos", "BOS"),
        ("eos", "IM_END / EOS"),
        ("pad", "PAD"),
        ("mask", "DLLM mask token id"),
        ("newline", "Newline token"),
        ("chat", "Chat-template special token"),
    ]
    parts = ["<div class='special-summary'>"]
    for key, label in labels:
        hits = positions.get(key, [])
        if hits:
            shown = ", ".join(str(v) for v in hits[:limit])
            tail = "" if len(hits) <= limit else f", ... +{len(hits) - limit} more"
            body = shown + tail
        else:
            body = "none"
        parts.append(f"<div><b>{label} positions:</b> {html.escape(body)}</div>")
    parts.append("</div>")
    return "".join(parts)


def _render_legend(items: list[tuple[str, str, str]]) -> str:
    parts = ["<div class='legend'>"]
    for fill, border, label in items:
        parts.append(
            "<span class='legend-item'>"
            f"<span class='legend-swatch' style='background:{fill};border:{border};'></span>"
            f"<span>{html.escape(label)}</span>"
            "</span>"
        )
    parts.append("</div>")
    return "".join(parts)


def _build_token_items(
    tokenizer: Any,
    token_ids: list[int],
    label_ids: list[int],
    *,
    block_size: int,
    mask_token_id: int,
    pad_token_id: int | None,
    mode: str,
    mask_row: list[bool] | None = None,
    masked_color: str = "#ef4444",
    visible_color: str = "#d1d5db",
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    mask_decoded_token = _get_token_display(tokenizer, int(mask_token_id))
    mask_token_piece = _get_token_piece(tokenizer, int(mask_token_id))
    safe_block_size = max(block_size, 1)
    for idx, (token_id, label_id) in enumerate(zip(token_ids, label_ids)):
        decoded_token = _get_token_display(tokenizer, int(token_id))
        token_piece = _get_token_piece(tokenizer, int(token_id))
        special_kind = _get_special_token_kind(
            tokenizer,
            int(token_id),
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )
        semantic_role = (
            _special_semantic_role(special_kind, decoded_token)
            if special_kind
            else "ordinary"
        )
        is_masked = bool(mask_row[idx]) if mask_row is not None and idx < len(mask_row) else False
        if mode == "supervision":
            if special_kind:
                color = _special_fill(special_kind)
            else:
                color = "#10b981" if int(label_id) != -100 else "#d1d5db"
        elif mode == "mask":
            if special_kind:
                color = _special_fill(special_kind)
            else:
                color = masked_color if is_masked else visible_color
        else:
            color = visible_color
        effective_token_id = int(mask_token_id) if is_masked else int(token_id)
        effective_decoded_token = mask_decoded_token if is_masked else decoded_token
        effective_token_piece = mask_token_piece if is_masked else token_piece
        items.append(
            {
                "color": color,
                "index": idx,
                "block_index": idx // safe_block_size,
                "token_id": int(token_id),
                "semantic_role": semantic_role,
                "effective_token_id": effective_token_id,
                "effective_decoded_token": effective_decoded_token,
                "effective_token_piece": effective_token_piece,
                "decoded_token": decoded_token,
                "token_piece": token_piece,
                "label_id": int(label_id),
                "supervised": int(label_id) != -100,
                "special_kind": special_kind or "ordinary",
                "masked": is_masked,
            }
        )
    return items


def _build_block_items(
    tokenizer: Any,
    token_ids: list[int],
    label_ids: list[int],
    mask_row: list[bool],
    *,
    block_size: int,
    mask_token_id: int,
    pad_token_id: int | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    safe_block_size = max(block_size, 1)
    num_blocks = (len(token_ids) + safe_block_size - 1) // safe_block_size
    for block_idx in range(num_blocks):
        start = block_idx * safe_block_size
        end = min(start + safe_block_size, len(token_ids))
        block_tokens = token_ids[start:end]
        block_labels = label_ids[start:end]
        block_mask = mask_row[start:end]
        special_kinds = [
            _get_special_token_kind(
                tokenizer,
                int(tok),
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
            )
            or "ordinary"
            for tok in block_tokens
        ]
        items.append(
            {
                "color": "#f59e0b" if any(bool(v) for v in block_mask) else "#d1d5db",
                "block_index": block_idx,
                "token_range": f"{start}:{end}",
                "masked_token_count": sum(1 for v in block_mask if bool(v)),
                "special_tokens": [kind for kind in special_kinds if kind != "ordinary"],
                "decoded_tokens": [_get_token_display(tokenizer, int(tok)) for tok in block_tokens],
                "token_ids": [int(tok) for tok in block_tokens],
                "label_ids": [int(lbl) for lbl in block_labels],
            }
        )
    return items


def _render_responsive_linear_canvas(
    *,
    canvas_id: str,
    items: list[dict[str, Any]],
    block_size: int,
    default_message: str,
    target_cell: int = 9,
    cell_height: int = 14,
    emphasize_special: bool = False,
    initial_index: int | None = None,
) -> str:
    safe_block_size = max(block_size, 1)
    return (
        f"<div class='canvas-widget'><div class='canvas-host' id='{canvas_id}_host'>"
        f"<canvas id='{canvas_id}' style='display:block;'></canvas></div>"
        "<script>"
        "(function(){"
        f"const items={_to_inline_json(items)};"
        f"const blockSize={safe_block_size};"
        f"const targetCell={target_cell};"
        f"const cellHeight={cell_height};"
        f"const emphasizeSpecial={str(emphasize_special).lower()};"
        f"let selected={json.dumps(initial_index)};"
        f"const host=document.getElementById('{canvas_id}_host');"
        f"const canvas=document.getElementById('{canvas_id}');"
        "const popup=document.createElement('pre'); popup.className='canvas-popup'; document.body.appendChild(popup);"
        "let cols=0; let rows=0;"
        "function format(item){const lines=[]; for(const [k,v] of Object.entries(item)){ if(k==='color') continue; lines.push(`${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`);} return lines.join('\\n'); }"
        "function measure(){ let c=Math.floor((host.clientWidth-16)/(targetCell+1)); c=Math.max(blockSize,c); c-=c%blockSize; if(c<blockSize) c=blockSize; return c; }"
        "function hidePopup(){ popup.classList.remove('visible'); }"
        "function showPopup(ev, text){ popup.textContent=text; popup.classList.add('visible'); popup.style.left='12px'; popup.style.top='12px'; const rect=popup.getBoundingClientRect(); let left=ev.clientX+12; let top=ev.clientY+12; if(left+rect.width>window.innerWidth-12){ left=Math.max(12, window.innerWidth-rect.width-12); } if(top+rect.height>window.innerHeight-12){ top=Math.max(12, ev.clientY-rect.height-12); } popup.style.left=`${left}px`; popup.style.top=`${top}px`; }"
        "function draw(){ const dpr=window.devicePixelRatio||1; cols=measure(); rows=Math.ceil(items.length/cols); const gap=1; const pad=8; const cssW=Math.max(120, cols*(targetCell+gap)+pad*2); const cssH=Math.max(60, rows*(cellHeight+gap)+pad*2); canvas.style.width=`${cssW}px`; canvas.style.height=`${cssH}px`; canvas.width=Math.ceil(cssW*dpr); canvas.height=Math.ceil(cssH*dpr); const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,cssW,cssH); ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,cssW,cssH); for(let idx=0; idx<items.length; idx++){ const row=Math.floor(idx/cols); const col=idx%cols; const x=pad+col*(targetCell+gap); const y=pad+row*(cellHeight+gap); const item=items[idx]; ctx.fillStyle=item.color; ctx.fillRect(x,y,targetCell,cellHeight); ctx.strokeStyle='white'; ctx.lineWidth=0.5; ctx.strokeRect(x,y,targetCell,cellHeight); if(emphasizeSpecial && item.special_kind && item.special_kind!=='ordinary'){ ctx.strokeStyle='#111827'; ctx.lineWidth=1.5; ctx.strokeRect(x+0.75,y+0.75,targetCell-1.5,cellHeight-1.5); if(targetCell>=10 && cellHeight>=12){ const label=item.special_kind==='bos'?'B':(item.special_kind==='eos'?'E':(item.special_kind==='pad'?'P':(item.special_kind==='mask'?'M':(item.special_kind==='newline'?'N':'T')))); ctx.fillStyle=(item.special_kind==='pad')?'#f9fafb':'#111827'; ctx.font='bold 9px sans-serif'; ctx.fillText(label,x+2,y+10); } } } if(blockSize>1){ ctx.strokeStyle='#0f172a'; ctx.lineWidth=0.8; for(let row=0; row<rows; row++){ for(let colStart=0; colStart<cols; colStart+=blockSize){ const startIdx=row*cols+colStart; if(startIdx>=items.length) break; const widthCells=Math.min(blockSize, cols-colStart, items.length-startIdx); const x=pad+colStart*(targetCell+gap)-0.5; const y=pad+row*(cellHeight+gap)-0.5; const w=widthCells*targetCell+Math.max(widthCells-1,0)*gap+1; const h=cellHeight+1; ctx.strokeRect(x,y,w,h); } } } if(selected!==null && selected>=0 && selected<items.length){ const row=Math.floor(selected/cols); const col=selected%cols; const x=pad+col*(targetCell+gap)-1; const y=pad+row*(cellHeight+gap)-1; ctx.strokeStyle='#111827'; ctx.lineWidth=2; ctx.strokeRect(x,y,targetCell+2,cellHeight+2);} }"
        "function hit(ev){ const rect=canvas.getBoundingClientRect(); const scaleX=canvas.width/rect.width; const scaleY=canvas.height/rect.height; const x=(ev.clientX-rect.left)*scaleX/(window.devicePixelRatio||1)-8; const y=(ev.clientY-rect.top)*scaleY/(window.devicePixelRatio||1)-8; if(x<0||y<0) return -1; const col=Math.floor(x/(targetCell+1)); const row=Math.floor(y/(cellHeight+1)); if(col<0||col>=cols||row<0) return -1; const idx=row*cols+col; return idx<items.length?idx:-1; }"
        "canvas.addEventListener('click', (ev)=>{ const idx=hit(ev); if(idx<0) return; selected=idx; const text=format(items[idx]); showPopup(ev, text); draw(); ev.stopPropagation(); });"
        "popup.addEventListener('click', (ev)=>{ ev.stopPropagation(); });"
        "document.addEventListener('click', (ev)=>{ if(ev.target!==canvas && !popup.contains(ev.target)){ hidePopup(); } });"
        "new ResizeObserver(draw).observe(host); window.addEventListener('resize', ()=>{ hidePopup(); draw(); }); window.addEventListener('scroll', hidePopup, true); draw();"
        "})();"
        "</script></div>"
    )


def _render_responsive_attention_canvas(
    *,
    canvas_id: str,
    row_bits: list[str],
    boundary_offset: int,
    block_size: int,
    logical_span: tuple[int, int],
    default_message: str,
    first_stream_label: str = "x0",
    second_stream_label: str = "xt",
) -> str:
    return (
        f"<div class='canvas-widget'><div class='canvas-host' id='{canvas_id}_host'>"
        f"<canvas id='{canvas_id}' style='display:block;'></canvas></div>"
        "<script>"
        "(function(){"
        f"const rows={_to_inline_json(row_bits)};"
        f"const boundary={boundary_offset};"
        f"const blockSize={max(block_size, 1)};"
        f"const logicalStart={logical_span[0]};"
        f"const logicalEnd={logical_span[1]};"
        f"const firstStream={json.dumps(first_stream_label)};"
        f"const secondStream={json.dumps(second_stream_label)};"
        f"const host=document.getElementById('{canvas_id}_host');"
        f"const canvas=document.getElementById('{canvas_id}');"
        "const popup=document.createElement('pre'); popup.className='canvas-popup'; document.body.appendChild(popup);"
        "let cell=3; let selected=null;"
        "function size(){ return rows.length; }"
        "function formatMeta(r,c){ const allowed=rows[r][c]==='1'; const qStream=r<boundary?firstStream:secondStream; const kvStream=c<boundary?firstStream:secondStream; const qLocal=logicalStart + (r%boundary); const kvLocal=logicalStart + (c%boundary); const qBlock=Math.floor(qLocal/blockSize); const kvBlock=Math.floor(kvLocal/blockSize); return ['q_row: '+r,'kv_col: '+c,'allowed: '+allowed,'q_stream: '+qStream,'kv_stream: '+kvStream,'q_logical_token: '+qLocal,'kv_logical_token: '+kvLocal,'q_block: '+qBlock,'kv_block: '+kvBlock,`logical_span: ${logicalStart}:${logicalEnd}`].join('\\n'); }"
        "function hidePopup(){ popup.classList.remove('visible'); }"
        "function showPopup(ev, text){ popup.textContent=text; popup.classList.add('visible'); popup.style.left='12px'; popup.style.top='12px'; const rect=popup.getBoundingClientRect(); let left=ev.clientX+12; let top=ev.clientY+12; if(left+rect.width>window.innerWidth-12){ left=Math.max(12, window.innerWidth-rect.width-12); } if(top+rect.height>window.innerHeight-12){ top=Math.max(12, ev.clientY-rect.height-12); } popup.style.left=`${left}px`; popup.style.top=`${top}px`; }"
        "function draw(){ const dpr=window.devicePixelRatio||1; const n=size(); const maxCell=5; cell=Math.max(2, Math.min(maxCell, Math.floor((host.clientWidth-12)/n))); const css=n*cell; canvas.style.width=`${css}px`; canvas.style.height=`${css}px`; canvas.width=Math.ceil(css*dpr); canvas.height=Math.ceil(css*dpr); const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,css,css); for(let r=0;r<n;r++){ const row=rows[r]; for(let c=0;c<n;c++){ ctx.fillStyle=row[c]==='1'?'#2563eb':'#e5e7eb'; ctx.fillRect(c*cell,r*cell,cell,cell);} } ctx.strokeStyle='#0f172a'; ctx.lineWidth=1; for(let p=0;p<=n;p+=blockSize){ ctx.beginPath(); ctx.moveTo(p*cell,0); ctx.lineTo(p*cell,css); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0,p*cell); ctx.lineTo(css,p*cell); ctx.stroke(); } ctx.strokeStyle='#ef4444'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(boundary*cell,0); ctx.lineTo(boundary*cell,css); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0,boundary*cell); ctx.lineTo(css,boundary*cell); ctx.stroke(); if(selected){ ctx.strokeStyle='#111827'; ctx.lineWidth=2; ctx.strokeRect(selected.c*cell, selected.r*cell, cell, cell);} }"
        "function hit(ev){ const rect=canvas.getBoundingClientRect(); const x=ev.clientX-rect.left; const y=ev.clientY-rect.top; const c=Math.floor(x/cell); const r=Math.floor(y/cell); const n=size(); if(r<0||c<0||r>=n||c>=n) return null; return {r,c}; }"
        "canvas.addEventListener('click', (ev)=>{ const hitPos=hit(ev); if(!hitPos) return; selected=hitPos; showPopup(ev, formatMeta(hitPos.r, hitPos.c)); draw(); ev.stopPropagation(); });"
        "popup.addEventListener('click', (ev)=>{ ev.stopPropagation(); });"
        "document.addEventListener('click', (ev)=>{ if(ev.target!==canvas && !popup.contains(ev.target)){ hidePopup(); } });"
        "new ResizeObserver(draw).observe(host); window.addEventListener('resize', ()=>{ hidePopup(); draw(); }); window.addEventListener('scroll', hidePopup, true); draw();"
        "})();"
        "</script></div>"
    )


def _render_token_table(
    tokenizer: Any,
    token_ids: list[int],
    label_ids: list[int],
    *,
    limit: int = 256,
) -> str:
    rows = []
    capped = min(len(token_ids), limit)
    for idx in range(capped):
        token_id = int(token_ids[idx])
        label_id = int(label_ids[idx])
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{token_id}</td>"
            f"<td>{label_id}</td>"
            f"<td class='tok-piece'>{html.escape(_get_token_piece(tokenizer, token_id))}</td>"
            f"<td class='tok-piece'>{html.escape(_get_token_display(tokenizer, token_id))}</td>"
            "</tr>"
        )
    tail = ""
    if len(token_ids) > capped:
        tail = f"<p class='muted'>Only first {capped} positions shown. Total length: {len(token_ids)}.</p>"
    return (
        "<table><thead><tr><th>idx</th><th>token_id</th><th>label</th><th>piece</th><th>decoded</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + tail
    )


def _render_full_chunk_supervision_svg(
    token_ids: list[int],
    labels: list[int],
    *,
    tokenizer: Any,
    mask_token_id: int,
    pad_token_id: int | None,
    block_size: int,
    canvas_id: str,
) -> str:
    items = _build_token_items(
        tokenizer,
        token_ids,
        labels,
        block_size=block_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        mode="supervision",
    )
    initial_index = next(
        (idx for idx, item in enumerate(items) if item["special_kind"] == "pad"),
        next((idx for idx, item in enumerate(items) if item["special_kind"] != "ordinary"), None),
    )
    return _render_responsive_linear_canvas(
        canvas_id=canvas_id,
        items=items,
        block_size=block_size,
        default_message="Click a token cell to inspect supervision / special-token metadata.",
        emphasize_special=True,
        initial_index=initial_index,
    )


def _render_branch_assignment_mask_svg(
    token_ids: list[int],
    labels: list[int],
    primary_mask: list[bool],
    complementary_mask: list[bool],
    *,
    tokenizer: Any,
    mask_token_id: int,
    pad_token_id: int | None,
    block_size: int,
    canvas_id: str,
) -> str:
    items: list[dict[str, Any]] = []
    safe_block_size = max(block_size, 1)
    mask_decoded_token = _get_token_display(tokenizer, int(mask_token_id))
    mask_token_piece = _get_token_piece(tokenizer, int(mask_token_id))
    for idx, (token_id, label_id) in enumerate(zip(token_ids, labels)):
        decoded_token = _get_token_display(tokenizer, int(token_id))
        token_piece = _get_token_piece(tokenizer, int(token_id))
        special_kind = _get_special_token_kind(
            tokenizer,
            int(token_id),
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )
        primary = bool(primary_mask[idx]) if idx < len(primary_mask) else False
        complementary = bool(complementary_mask[idx]) if idx < len(complementary_mask) else False
        if special_kind:
            semantic_role = _special_semantic_role(special_kind, decoded_token)
        elif primary:
            semantic_role = "primary masked branch"
        elif complementary:
            semantic_role = "complementary masked branch"
        elif int(label_id) == -100:
            semantic_role = "ignored token (label=-100; not tokenizer PAD)"
        else:
            semantic_role = "visible token"
        if special_kind:
            color = _special_fill(special_kind)
        elif primary:
            color = "#ef4444"
        elif complementary:
            color = "#f97316"
        else:
            color = "#d1d5db"
        effectively_masked = primary or complementary
        items.append(
            {
                "color": color,
                "index": idx,
                "block_index": idx // safe_block_size,
                "token_id": int(token_id),
                "semantic_role": semantic_role,
                "effective_token_id": int(mask_token_id) if effectively_masked else int(token_id),
                "effective_decoded_token": mask_decoded_token if effectively_masked else decoded_token,
                "effective_token_piece": mask_token_piece if effectively_masked else token_piece,
                "decoded_token": decoded_token,
                "token_piece": token_piece,
                "label_id": int(label_id),
                "supervised": int(label_id) != -100,
                "special_kind": special_kind or "ordinary",
                "primary_masked": primary,
                "complementary_masked": complementary,
            }
        )
    initial_index = next(
        (idx for idx, item in enumerate(items) if item["primary_masked"]),
        next((idx for idx, item in enumerate(items) if item["complementary_masked"]), None),
    )
    return _render_responsive_linear_canvas(
        canvas_id=canvas_id,
        items=items,
        block_size=block_size,
        default_message="Click a token cell to inspect original token metadata and effective <|MASK|> replacement.",
        emphasize_special=True,
        initial_index=initial_index,
    )


def _render_full_block_mask_svg(
    token_ids: list[int],
    labels: list[int],
    mask_row: list[bool],
    *,
    tokenizer: Any,
    mask_token_id: int,
    pad_token_id: int | None,
    block_size: int,
    canvas_id: str,
) -> str:
    items = _build_block_items(
        tokenizer,
        token_ids,
        labels,
        mask_row,
        block_size=block_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
    )
    return _render_responsive_linear_canvas(
        canvas_id=canvas_id,
        items=items,
        block_size=1,
        default_message="Click a block cell to inspect block-range metadata.",
        target_cell=16,
        cell_height=16,
    )


def _compute_aligned_x0_xt_window(
    *,
    seq_len: int,
    block_size: int,
    logical_start: int,
    window: int = 128,
) -> tuple[list[str], int, tuple[int, int], tuple[str, str]] | None:
    try:
        from torchtitan.models.qwen3.model.model_dllm import block_diff_mask
    except Exception:
        return None

    logical_end = min(logical_start + window, seq_len)
    logical_start = max(0, logical_end - window)
    x0_idx = torch.arange(logical_start, logical_end, dtype=torch.long) + seq_len
    xt_idx = torch.arange(logical_start, logical_end, dtype=torch.long)
    idx = torch.cat([x0_idx, xt_idx], dim=0)
    q_mesh, kv_mesh = torch.meshgrid(idx, idx, indexing="ij")
    mask = block_diff_mask(
        torch.zeros_like(q_mesh),
        torch.zeros_like(q_mesh),
        q_mesh,
        kv_mesh,
        block_size=block_size,
        n=seq_len,
    )
    rows = ["".join("1" if bool(v) else "0" for v in row.tolist()) for row in mask]
    return rows, logical_end - logical_start, (logical_start, logical_end), ("x0", "xt")


def _render_attention_window_canvas(
    *,
    row_bits: list[str],
    boundary_offset: int,
    block_size: int,
    logical_span: tuple[int, int],
    title: str,
    note: str,
    canvas_id: str,
    first_stream_label: str = "x0",
    second_stream_label: str = "xt",
) -> str:
    return (
        f"<div><p><b>{html.escape(title)}</b></p>"
        f"<p class='muted'>{html.escape(note)}</p>"
        + _render_responsive_attention_canvas(
            canvas_id=canvas_id,
            row_bits=row_bits,
            boundary_offset=boundary_offset,
            block_size=block_size,
            logical_span=logical_span,
            default_message="Click a matrix cell to inspect q/kv stream, token index, block index, and attention state.",
            first_stream_label=first_stream_label,
            second_stream_label=second_stream_label,
        )
        + "</div>"
    )


def _extract_branch_masks(
    payload: dict[str, Any],
    *,
    sample_id: int,
    seq_len: int,
) -> tuple[list[bool], list[bool], list[tuple[str, list[bool]]]]:
    primary = [False] * seq_len
    complementary = [False] * seq_len
    extras: list[tuple[str, list[bool]]] = []
    viz_payload = payload.get("viz_payload")
    if not isinstance(viz_payload, dict):
        return primary, complementary, extras
    sample_branch_masks = viz_payload.get("sample_branch_masks")
    if not isinstance(sample_branch_masks, dict):
        return primary, complementary, extras
    serialized_branches = sample_branch_masks.get(str(sample_id), [])
    if not isinstance(serialized_branches, list):
        return primary, complementary, extras
    for branch_entry in serialized_branches:
        if not isinstance(branch_entry, dict):
            continue
        branch_name = branch_entry.get("branch")
        mask = branch_entry.get("mask")
        if not isinstance(branch_name, str) or not isinstance(mask, list):
            continue
        normalized = [bool(v) for v in mask[:seq_len]]
        if len(normalized) < seq_len:
            normalized.extend([False] * (seq_len - len(normalized)))
        if branch_name == "primary":
            primary = normalized
        elif branch_name == "complementary":
            complementary = normalized
        else:
            extras.append((branch_name, normalized))
    if not any(primary) and extras:
        primary = extras[0][1]
        extras = extras[1:]
    return primary, complementary, extras


def build_training_step_preview_html(
    *,
    tokenizer: Any,
    payload: dict[str, Any],
    block_size: int,
    dllm_layout: str,
    mask_token_id: int,
    pad_token_id: int | None,
    reference_html_url: str | None = None,
) -> str:
    samples = payload.get("samples", [])
    step = int(payload.get("step", 0))
    seq_len = len(samples[0].get("input_ids", [])) if samples else 0
    viz_payload = payload.get("viz_payload")
    target_slice = None
    if isinstance(viz_payload, dict):
        maybe_target_slice = viz_payload.get("target_slice")
        if (
            isinstance(maybe_target_slice, list)
            and len(maybe_target_slice) == 2
            and all(isinstance(x, int) for x in maybe_target_slice)
        ):
            target_slice = (maybe_target_slice[0], maybe_target_slice[1])

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Training Data Visualization Step {step}</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>Training Data Visualization Step {step}</h1>",
        (
            "<p class='section-note'>This page renders the actual captured training batch for this step: "
            "token ids, supervision labels, sampled DLLM branch masks, and the aligned block-attention window. "
            "Unlike the fixed preview, every step on this page is driven by the current batch payload.</p>"
        ),
        "<div class='card'>",
        f"<span class='pill'>step: {step}</span>",
        f"<span class='pill'>samples: {len(samples)}</span>",
        f"<span class='pill'>seq_len: {seq_len}</span>",
        f"<span class='pill'>block_size: {block_size}</span>",
        f"<span class='pill'>layout: {html.escape(dllm_layout or 'unknown')}</span>",
    ]
    if target_slice is not None:
        parts.append(
            f"<span class='pill'>target_slice: {target_slice[0]}:{target_slice[1]}</span>"
        )
    if reference_html_url:
        escaped_reference_url = html.escape(reference_html_url, quote=True)
        parts.append(
            f"<span class='pill'>reference: <a href='{escaped_reference_url}' target='_blank' rel='noopener noreferrer'>rich preview template</a></span>"
        )
    parts.append("</div>")

    for sample_payload in samples:
        sample_id = int(sample_payload.get("sample_id", 0))
        token_ids = [int(token_id) for token_id in sample_payload.get("input_ids", [])]
        label_ids = [int(label) for label in sample_payload.get("labels", [])]
        supervised_tokens = sum(1 for label in label_ids if label != -100)
        special_counts = _count_special_tokens(
            tokenizer,
            token_ids,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )
        special_positions = _collect_special_token_positions(
            tokenizer,
            token_ids,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )
        primary_mask, complementary_mask, extra_masks = _extract_branch_masks(
            payload,
            sample_id=sample_id,
            seq_len=len(token_ids),
        )
        masked_anchor = next(
            (
                idx
                for idx, (primary, complementary) in enumerate(zip(primary_mask, complementary_mask))
                if primary or complementary
            ),
            next((idx for idx, label in enumerate(label_ids) if label != -100), 0),
        )
        safe_block_size = max(block_size, 1)
        aligned_start = max(
            0,
            min(
                max(len(token_ids) - 128, 0),
                (masked_anchor // safe_block_size) * safe_block_size - 64,
            ),
        )
        attention_html = ""
        if dllm_layout == "x0_xt_doubled" and safe_block_size > 0 and token_ids:
            attention_data = _compute_aligned_x0_xt_window(
                seq_len=len(token_ids),
                block_size=safe_block_size,
                logical_start=aligned_start,
                window=min(128, len(token_ids)),
            )
            if attention_data is not None:
                rows, boundary_offset, logical_span, (first_stream_label, second_stream_label) = attention_data
                attention_html = _render_attention_window_canvas(
                    row_bits=rows,
                    boundary_offset=boundary_offset,
                    block_size=safe_block_size,
                    logical_span=logical_span,
                    title="Aligned x0/xt Attention Window",
                    note=(
                        "This attention view is aligned around the first active masked token in the current batch sample, "
                        "so the block pattern changes with the step payload."
                    ),
                    canvas_id=f"step_{step}_sample_{sample_id}_attention",
                    first_stream_label=first_stream_label,
                    second_stream_label=second_stream_label,
                )

        input_text = html.escape(tokenizer.decode(token_ids))
        label_text = html.escape(
            tokenizer.decode([token for token, label in zip(token_ids, label_ids) if label != -100])
        )
        extra_branch_note = ""
        if extra_masks:
            extra_branch_note = " Extra branches: " + ", ".join(
                f"{html.escape(name)}={sum(1 for v in mask if v)}" for name, mask in extra_masks
            )
        parts.extend(
            [
                "<div class='card'>",
                f"<h2>Sample {sample_id}</h2>",
                (
                    "<p>"
                    f"<b>Token count:</b> {len(token_ids)} | "
                    f"<b>Supervised tokens:</b> {supervised_tokens} | "
                    f"<b>Primary masked:</b> {sum(1 for v in primary_mask if v)} | "
                    f"<b>Complementary masked:</b> {sum(1 for v in complementary_mask if v)}"
                    f"{extra_branch_note}</p>"
                ),
                (
                    "<p class='muted'>"
                    f"Counts in this step sample: BOS={special_counts['bos']}, "
                    f"IM_END/EOS={special_counts['eos']}, PAD={special_counts['pad']}, "
                    f"raw input mask-token-id hits={special_counts['mask']}, "
                    f"newline tokens={special_counts['newline']}, "
                    f"chat-template specials={special_counts['chat']}."
                    "</p>"
                ),
                _render_special_position_summary(special_positions),
                _render_legend(
                    [
                        ("#10b981", "1px solid #d1d5db", "supervised assistant token"),
                        ("#ef4444", "1px solid #d1d5db", "masked token in primary branch"),
                        ("#f97316", "1px solid #d1d5db", "masked token in complementary branch"),
                        ("#d1d5db", "1px solid #d1d5db", "ignored / visible token"),
                        ("#8b5cf6", "1px solid #d1d5db", "BOS"),
                        ("#f59e0b", "1px solid #d1d5db", "IM_END / EOS"),
                        ("#4b5563", "1px solid #d1d5db", "PAD"),
                        ("#d946ef", "1px solid #d1d5db", "DLLM mask token id"),
                        ("#14b8a6", "1px solid #d1d5db", "newline token"),
                        ("#06b6d4", "1px solid #d1d5db", "chat-template special token"),
                        ("transparent", "1.5px solid #0f172a", f"one DLLM chunk block ({safe_block_size} tokens)"),
                    ]
                ),
                "<div class='compare-grid'>",
                "<div class='compare-panel'>",
                "<h3>Supervision Token Overview (full)</h3>",
                "<div class='scroll-pane'>"
                + _render_full_chunk_supervision_svg(
                    token_ids,
                    label_ids,
                    tokenizer=tokenizer,
                    mask_token_id=mask_token_id,
                    pad_token_id=pad_token_id,
                    block_size=safe_block_size,
                    canvas_id=f"step_{step}_sample_{sample_id}_supervision",
                )
                + "</div>",
                "</div>",
                "<div class='compare-panel'>",
                "<h3>DLLM Branch Assignment Mask (full)</h3>",
                "<div class='scroll-pane'>"
                + _render_branch_assignment_mask_svg(
                    token_ids,
                    label_ids,
                    primary_mask,
                    complementary_mask,
                    tokenizer=tokenizer,
                    mask_token_id=mask_token_id,
                    pad_token_id=pad_token_id,
                    block_size=safe_block_size,
                    canvas_id=f"step_{step}_sample_{sample_id}_branch_assignment",
                )
                + "</div>",
                "</div>",
                "</div>",
                "<h3>DLLM Block Mask</h3>",
                _render_legend(
                    [
                        ("#f59e0b", "1px solid #d1d5db", "block contains at least one primary-branch masked token"),
                        ("#d1d5db", "1px solid #d1d5db", "block has no primary-branch masked token"),
                    ]
                ),
                "<div class='scroll-pane'>"
                + _render_full_block_mask_svg(
                    token_ids,
                    label_ids,
                    primary_mask,
                    tokenizer=tokenizer,
                    mask_token_id=mask_token_id,
                    pad_token_id=pad_token_id,
                    block_size=safe_block_size,
                    canvas_id=f"step_{step}_sample_{sample_id}_block_mask",
                )
                + "</div>",
            ]
        )
        if attention_html:
            parts.extend(["<h3>DLLM Attention</h3>", attention_html])
        parts.extend(
            [
                "<div class='grid'>",
                "<div>",
                "<h3>Input</h3>",
                f"<div class='scroll-pane'><pre>{input_text}</pre></div>",
                "</div>",
                "<div>",
                "<h3>Supervised Label Text</h3>",
                f"<div class='scroll-pane'><pre>{label_text}</pre></div>",
                "</div>",
                "</div>",
                "<h3>Tokenized + Labeled Table</h3>",
                "<div class='scroll-pane'>"
                + _render_token_table(tokenizer, token_ids, label_ids, limit=256)
                + "</div>",
                "</div>",
            ]
        )

    parts.append("</body></html>")
    return "".join(parts)
