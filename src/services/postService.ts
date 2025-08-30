import matter from 'gray-matter';
import type { Post, TreeNode } from '../types/index';

let allPosts: Post[] | null = null;

export function getAllPosts(): Post[] {
  if (allPosts) {
    return allPosts;
  }

  const modules = import.meta.glob('../posts/**/*.{md,markdown}', { query: '?raw', import: 'default', eager: true });
  const posts: Post[] = Object.entries(modules).map(([filepath, rawContent]) => {
    const { data, content } = matter(rawContent as string);
    // 从 '../posts/2023/my-first-post.md' 提取 '2023/my-first-post'
    const path = filepath.replace('../posts/', '').replace(/\.(md|markdown)$/, '');
    return { path, frontmatter: data, content };
  }).filter(post => !post.frontmatter.hidden);

  allPosts = posts;
  return posts;
}

// 将扁平的文章列表转换为树形结构
export function buildDirectoryTree(): TreeNode[] {
  const posts = getAllPosts();
  const tree: TreeNode[] = [];
  const directoryMap = new Map<string, TreeNode>();

  // 创建一个从path到post的映射，用于获取文章的日期信息
  const postMap = new Map<string, Post>();
  posts.forEach(post => {
    postMap.set(post.path, post);
  });

  posts.forEach(post => {
    const pathParts = post.path.split('/');

    // 如果是根目录文件（如 introduction.md）
    if (pathParts.length === 1) {
      tree.push({
        name: post.frontmatter.title || pathParts[0],
        path: post.path,
        isDirectory: false
      });
      return;
    }

    // 处理有目录的文件
    let currentPath = '';
    let currentLevel = tree;

    // 遍历路径的每一部分，除了最后的文件名
    for (let i = 0; i < pathParts.length - 1; i++) {
      const dirName = pathParts[i];
      currentPath = currentPath ? `${currentPath}/${dirName}` : dirName;

      // 查找或创建目录节点
      let dirNode = currentLevel.find(node => node.name === dirName && node.isDirectory);

      if (!dirNode) {
        dirNode = {
          name: dirName,
          path: currentPath, // 添加目录的完整路径
          children: [],
          isDirectory: true
        };
        currentLevel.push(dirNode);
        directoryMap.set(currentPath, dirNode);
      }

      currentLevel = dirNode.children!;
    }

    // 添加文件节点
    const fileName = pathParts[pathParts.length - 1];
    currentLevel.push({
      name: post.frontmatter.title || fileName,
      path: post.path,
      isDirectory: false
    });
  });

  // 获取文章的日期，用于排序
  function getPostDate(node: TreeNode): Date {
    if (node.isDirectory) {
      return new Date(0); // 目录使用最早日期，确保排在前面
    }

    // 确保path存在
    if (!node.path) {
      return new Date(0);
    }

    const post = postMap.get(node.path);
    if (post && post.frontmatter.date) {
      // 处理不同的日期格式
      const dateStr = post.frontmatter.date;
      if (typeof dateStr === 'string') {
        // 支持 '2023-07-21 11:00' 和 '2024-01-01' 格式
        return new Date(dateStr);
      }
    }

    // 如果没有日期，尝试从文件名中提取日期（如 2023-07-22-xxx.md）
    const match = node.path.match(/(\d{4}-\d{2}-\d{2})/);
    if (match) {
      return new Date(match[1]);
    }

    // 如果都没有，使用最早日期
    return new Date(0);
  }

  // 对每层进行排序：目录在前，文件按日期倒序排列
  function sortTree(nodes: TreeNode[]): TreeNode[] {
    return nodes
      .sort((a, b) => {
        // 目录在前，文件在后
        if (a.isDirectory !== b.isDirectory) {
          return a.isDirectory ? -1 : 1;
        }

        if (a.isDirectory && b.isDirectory) {
          // 目录之间按名称排序（可以考虑按年份倒序）
          // 如果目录名是年份，按年份倒序排列
          const aYear = parseInt(a.name);
          const bYear = parseInt(b.name);
          if (!isNaN(aYear) && !isNaN(bYear)) {
            return bYear - aYear; // 年份倒序
          }
          return a.name.localeCompare(b.name);
        }

        // 文件之间按日期倒序排列（最新的在前）
        const dateA = getPostDate(a);
        const dateB = getPostDate(b);
        return dateB.getTime() - dateA.getTime();
      })
      .map(node => ({
        ...node,
        children: node.children ? sortTree(node.children) : undefined
      }));
  }

  return sortTree(tree);
}

// 保留原来的函数用于向后兼容
export function loadPosts() {
  const posts = getAllPosts();
  return {
    count: posts.length,
    paths: posts.map(p => p.path),
    posts: posts
  };
}
