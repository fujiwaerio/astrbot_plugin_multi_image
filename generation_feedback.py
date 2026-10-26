"""Read bounded image headers and describe results without resizing or new dependencies."""

from __future__ import annotations

HEADER_LIMIT = 256 * 1024
_JPEG_SOF = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Inspect PNG, JPEG, WebP, GIF or BMP headers; unknown/truncated data stays unknown."""
    data = data[:HEADER_LIMIT]
    dimensions = _header_dimensions(data)
    if dimensions and min(dimensions) > 0:
        return dimensions
    return None


def _header_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if len(data) >= 10 and data[:6] in {b"GIF87a", b"GIF89a"}:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if (
        len(data) >= 26
        and data[:2] == b"BM"
        and int.from_bytes(data[14:18], "little") >= 40
    ):
        return int.from_bytes(data[18:22], "little", signed=True), abs(
            int.from_bytes(data[22:26], "little", signed=True)
        )
    if len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        if data[12:16] == b"VP8X":
            return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(
                data[27:30], "little"
            )
        if data[12:16] == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            return int.from_bytes(data[26:28], "little") & 0x3FFF, int.from_bytes(
                data[28:30], "little"
            ) & 0x3FFF
    if (
        len(data) >= 25
        and data[:4] == b"RIFF"
        and data[8:16] == b"WEBPVP8L"
        and data[20] == 0x2F
    ):
        bits = int.from_bytes(data[21:25], "little")
        return 1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF)
    if data[:2] == b"\xff\xd8":
        offset = 2
        while offset + 1 < len(data):
            if data[offset] != 0xFF:
                return None
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            if offset >= len(data):
                return None
            marker = data[offset]
            offset += 1
            if marker in {0xD9, 0xDA}:
                return None
            if marker in {0x01, 0xD8, *range(0xD0, 0xD8)}:
                continue
            if offset + 2 > len(data):
                return None
            length = int.from_bytes(data[offset : offset + 2], "big")
            if length < 2 or offset + length > len(data):
                return None
            if marker in _JPEG_SOF and length >= 8:
                return int.from_bytes(
                    data[offset + 5 : offset + 7], "big"
                ), int.from_bytes(data[offset + 3 : offset + 5], "big")
            offset += length
    return None


def result_message(
    dimensions: list[tuple[int, int] | None],
    expected: tuple[int, int] | None,
    elapsed: float,
) -> str:
    lines = [f"生成完成：{len(dimensions)} 张，总耗时 {max(0.0, elapsed):.1f} 秒。"]
    for index, actual in enumerate(dimensions, 1):
        if actual is None:
            lines.append(f"第 {index} 张：尺寸未能核验。")
            continue
        detail = f"第 {index} 张：实际 {actual[0]}×{actual[1]}"
        if expected and actual != expected:
            detail += f"；请求 {expected[0]}×{expected[1]}，上游返回尺寸与请求不一致"
        lines.append(detail + "。")
    return "\n".join(lines)
