import DefaultTheme from 'vitepress/theme'
import { onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()

    function renderMath() {
      if (typeof window !== 'undefined' && (window as any).renderMathInElement) {
        nextTick(() => {
          (window as any).renderMathInElement(document.body, {
            delimiters: [
              { left: '$$', right: '$$', display: true },
              { left: '$', right: '$', display: false },
            ],
            throwOnError: false,
          })
        })
      }
    }

    onMounted(() => {
      // Wait for KaTeX to load, then render
      const check = setInterval(() => {
        if (typeof (window as any).renderMathInElement !== 'undefined') {
          clearInterval(check)
          renderMath()
        }
      }, 100)
      // Stop checking after 5 seconds
      setTimeout(() => clearInterval(check), 5000)
    })

    watch(() => route.path, () => renderMath())
  },
}
