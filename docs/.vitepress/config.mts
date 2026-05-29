import { defineConfig } from 'vitepress'
import sidebar from './sidebar'

// https://vitepress.dev/zh/reference/site-config
export default defineConfig({
  base: '/llmtech/',
  lang: 'zh-CN',
  title: "Robin's AI Lab",
  description: '探索人工智能的无限可能 — 大语言模型、强化学习、深度学习等前沿AI技术的研究与实践',

  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: true,

  head: [
    ['link', { rel: 'icon', href: '/llmtech/favicon.ico' }],
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css' }],
    ['script', { defer: '', src: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js' }],
    ['script', { defer: '', src: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' }],
    ['script', {}, `
      function renderKaTeX() {
        if (typeof renderMathInElement === 'undefined') {
          setTimeout(renderKaTeX, 100);
          return;
        }
        renderMathInElement(document.body, {
          delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
          ],
          throwOnError: false
        });
      }
      document.addEventListener('DOMContentLoaded', renderKaTeX);
      document.addEventListener('vitepress:afterRouteChange', renderKaTeX);
    `]
  ],

  markdown: {
    lineNumbers: true,
    container: {
      tipLabel: '提示',
      warningLabel: '警告',
      dangerLabel: '危险',
      infoLabel: '信息',
      detailsLabel: '详情'
    }
  },

  themeConfig: {
    // https://vitepress.dev/zh/reference/default-theme-config
    search: {
      provider: 'local'
    },

    nav: [
      { text: '首页', link: '/' },
      {
        text: 'AI 技术',
        items: [
          { text: '🦙 大语言模型', link: '/aigc/' },
          { text: '🤗 强化微调', link: '/rlhf/' },
          { text: '✌️ 推理部署', link: '/inference/' },
          { text: '🏗️ 模型架构', link: '/architecture/' },
          { text: '🚀 大规模并行', link: '/ultrascale/' },
          { text: '🍭 扩散模型', link: '/diffusion/' },
          { text: '🍔 多模态', link: '/multimodal/' },
          { text: '🍒 强化学习', link: '/rlwiki/' }
        ]
      },
      { text: '🇨🇳 Ascend生态', link: '/ascend/' },
      {
        text: '更多',
        items: [
          { text: '⚒️ 效率工具', link: '/toolbox/' },
          { text: '📖 深度学习百科', link: '/dlwiki/' },
          { text: '🤖 机器学习百科', link: '/mlwiki/' },
          { text: '🔗 外部链接', link: '/links' }
        ]
      }
    ],

    sidebar,

    socialLinks: [
      { icon: 'github', link: 'https://github.com/jianzhnie' }
    ],

    footer: {
      message: 'Maintained by Robin',
      copyright: '© 2017 — 2025. Powered by VitePress'
    },

    outline: {
      level: [2, 3],
      label: '文章目录'
    },

    editLink: {
      pattern: 'https://github.com/jianzhnie/llmtech/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: { dateStyle: 'short', timeStyle: 'short' }
    },

    notFound: {
      title: '页面未找到',
      quote: '您访问的页面不存在',
      linkLabel: '返回首页',
      linkText: '返回首页'
    },

    darkModeSwitchLabel: '主题切换',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '回到顶部'
  }
})
