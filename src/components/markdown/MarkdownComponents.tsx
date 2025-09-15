import type { Components } from 'react-markdown';
import CodeBlock from './CodeBlock';
import PlainCodeBlock from './PlainCodeBlock';
import MermaidBlock from './MermaidBlock';

export const markdownComponents: Components = {
  // 标题组件 - 移动端优化
  h1: ({ children }) => <h1 className="text-2xl md:text-3xl font-bold mb-3 md:mb-4 break-words">{children}</h1>,
  h2: ({ children }) => <h2 className="text-xl md:text-2xl font-bold mb-2 md:mb-3 break-words">{children}</h2>,
  h3: ({ children }) => <h3 className="text-lg md:text-xl font-bold mb-2 break-words">{children}</h3>,
  h4: ({ children }) => <h4 className="text-base md:text-lg font-bold mb-2 break-words">{children}</h4>,
  h5: ({ children }) => <h5 className="text-sm md:text-base font-bold mb-1 break-words">{children}</h5>,
  h6: ({ children }) => <h6 className="text-sm font-bold mb-1 break-words">{children}</h6>,

  // 段落和列表 - 移动端优化
  p: ({ children }) => <p className="mb-3 md:mb-4 leading-relaxed break-words">{children}</p>,
  ul: ({ children }) => <ul className="list-disc list-inside mb-3 md:mb-4 space-y-1 pl-2">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal list-inside mb-3 md:mb-4 space-y-1 pl-2">{children}</ol>,
  li: ({ children }) => <li className="ml-2 md:ml-4 break-words">{children}</li>,

  // 引用块
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-gray-300 dark:border-gray-600 pl-3 md:pl-4 py-2 mb-3 md:mb-4 italic text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-slate-800 rounded-r">
      {children}
    </blockquote>
  ),

  // 表格 - 移动端优化
  table: ({ children }) => (
    <div className="overflow-x-auto mb-3 md:mb-4">
      <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-600 text-sm">
        {children}
      </table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-gray-300 dark:border-gray-600 bg-gray-100 dark:bg-slate-700 px-2 md:px-4 py-2 text-left font-semibold">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-gray-300 dark:border-gray-600 px-2 md:px-4 py-2 break-words">
      {children}
    </td>
  ),

  // 处理代码块（pre + code 组合）
  pre: ({ children }) => {
    // 检查是否包含 code 元素
    if (children && typeof children === 'object' && 'props' in children) {
      const codeElement = children as any;
      const className = codeElement.props?.className || '';
      const match = /language-(\w+)/.exec(className);
      const language = match ? match[1] : 'text';
      const codeContent = String(codeElement.props?.children || '').replace(/\n$/, '');

      // 检查是否是 Mermaid 图表
      if (language === 'mermaid') {
        return <MermaidBlock chart={codeContent} />;
      }

      return <CodeBlock language={language} code={codeContent} />;
    }

    // 如果不是标准的代码块结构，使用普通代码块
    const codeContent = String(children || '');
    return <PlainCodeBlock code={codeContent}>{children}</PlainCodeBlock>;
  },

  // 内联代码 - 移动端优化
  code: ({ children }) => (
    <code className="bg-gray-200 dark:bg-gray-700 px-1.5 py-0.5 rounded text-xs md:text-sm font-mono break-all">
      {children}
    </code>
  ),

  // 链接 - 移动端优化
  a: ({ href, children }) => (
    <a
      href={href}
      className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 underline break-all"
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </a>
  ),

  // 分割线
  hr: () => <hr className="border-gray-300 dark:border-gray-600 my-4 md:my-6" />,

  // 图片 - 移动端优化
  img: ({ src, alt }) => (
    <img
      src={src}
      alt={alt}
      className="max-w-full h-auto rounded-lg shadow-md my-3 md:my-4 mx-auto block"
      loading="lazy"
    />
  ),
};
