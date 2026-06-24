// Text chunking utility

export function chunkText(
  text: string,
  chunkSize: number = 500,
  chunkOverlap: number = 100,
  separators: string[] = ["\n\n", "\n", "。", ".", "！", "？", " ", ""]
): string[] {
  if (!text || text.trim().length === 0) return [];
  
  const chunks: string[] = [];
  let remaining = text;
  
  while (remaining.length > 0) {
    if (remaining.length <= chunkSize) {
      chunks.push(remaining.trim());
      break;
    }
    
    // Find the best split point
    let splitIndex = chunkSize;
    for (const sep of separators) {
      const pos = remaining.lastIndexOf(sep, chunkSize);
      if (pos > chunkSize * 0.3) {
        splitIndex = pos + sep.length;
        break;
      }
    }
    
    chunks.push(remaining.slice(0, splitIndex).trim());
    remaining = remaining.slice(Math.max(0, splitIndex - chunkOverlap));
  }
  
  return chunks.filter(c => c.length > 0);
}
