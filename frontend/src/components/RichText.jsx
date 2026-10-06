// Minimal, XSS-safe markdown subset: paragraphs, bullet/numbered lists, **bold**, `code`.

function renderInline(text, keyPrefix) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return parts.map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={key}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={key}>{part.slice(1, -1)}</code>;
    return <span key={key}>{part}</span>;
  });
}

const LIST_ITEM = /^\s*(?:[-*•]|\d+[.)])\s+(.*)$/;

export default function RichText({ text }) {
  const blocks = [];
  let list = null;

  text.split("\n").forEach((rawLine, idx) => {
    const line = rawLine.trimEnd();
    const match = line.match(LIST_ITEM);
    if (match) {
      const ordered = /^\s*\d/.test(line);
      if (!list || list.ordered !== ordered) {
        list = { type: "list", ordered, items: [] };
        blocks.push(list);
      }
      list.items.push(match[1]);
      return;
    }
    list = null;
    if (line.trim()) blocks.push({ type: "p", text: line.replace(/^#+\s*/, ""), idx });
  });

  return (
    <div className="rich-text">
      {blocks.map((block, i) => {
        if (block.type === "list") {
          const Tag = block.ordered ? "ol" : "ul";
          return (
            <Tag key={i}>
              {block.items.map((item, j) => (
                <li key={j}>{renderInline(item, `${i}-${j}`)}</li>
              ))}
            </Tag>
          );
        }
        return <p key={i}>{renderInline(block.text, `${i}`)}</p>;
      })}
    </div>
  );
}
