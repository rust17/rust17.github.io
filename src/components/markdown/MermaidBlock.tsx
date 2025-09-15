import { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

interface MermaidBlockProps {
  chart: string;
}

function MermaidBlock({ chart }: MermaidBlockProps) {
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // 初始化 mermaid
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: {
        useMaxWidth: false,
        htmlLabels: true,
        curve: 'basis',
      },
      gantt: {
        useMaxWidth: false,
      },
      sequence: {
        useMaxWidth: false,
      },
    });
  }, []);

  useEffect(() => {
    if (elementRef.current) {
      // 清空之前的内容
      elementRef.current.innerHTML = '';

      // 生成唯一的 ID
      const id = `mermaid-${Math.random().toString(36).substring(2, 11)}`;

      // 渲染 mermaid 图表
      mermaid.render(id, chart).then(({ svg }) => {
        if (elementRef.current) {
          elementRef.current.innerHTML = svg;
        }
      }).catch((error) => {
        console.error('Mermaid rendering error:', error);
        if (elementRef.current) {
          elementRef.current.innerHTML = `<div class="text-red-500 p-4 border border-red-300 rounded">
            <strong>Mermaid 渲染错误:</strong><br/>
            <pre class="mt-2 text-sm">${error.message || '未知错误'}</pre>
          </div>`;
        }
      });
    }
  }, [chart]);

  return (
    <div className="my-6 w-full">
      <div
        ref={elementRef}
        className="w-full overflow-x-auto overflow-y-auto max-h-[80vh] p-2 md:p-4"
        style={{
          minHeight: '200px',
          minWidth: '100%'
        }}
      />
    </div>
  );
}

export default MermaidBlock;
