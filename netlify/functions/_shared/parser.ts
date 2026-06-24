// Document parser for PDF, DOCX, MD, TXT

export async function parseFile(buffer: ArrayBuffer, filename: string, mimeType: string): Promise<string> {
  const ext = filename.toLowerCase().split('.').pop() || '';
  
  switch (ext) {
    case 'pdf':
      return await parsePdf(buffer);
    case 'docx':
    case 'doc':
      return await parseDocx(buffer);
    case 'md':
    case 'txt':
      return parseText(buffer);
    default:
      throw new Error(`Unsupported file type: ${ext}`);
  }
}

async function parsePdf(buffer: ArrayBuffer): Promise<string> {
  try {
    // Dynamic import to avoid bundling when not used
    const pdfParse = (await import("pdf-parse")).default;
    const data = Buffer.from(buffer);
    const result = await pdfParse(data);
    return result.text || '';
  } catch (e: any) {
    throw new Error(`PDF parse failed: ${e.message}`);
  }
}

async function parseDocx(buffer: ArrayBuffer): Promise<string> {
  try {
    const mammoth = await import("mammoth");
    const data = Buffer.from(buffer);
    const result = await mammoth.extractRawText({ buffer: data });
    return result.value || '';
  } catch (e: any) {
    throw new Error(`DOCX parse failed: ${e.message}`);
  }
}

function parseText(buffer: ArrayBuffer): string {
  const decoder = new TextDecoder('utf-8');
  return decoder.decode(buffer);
}
