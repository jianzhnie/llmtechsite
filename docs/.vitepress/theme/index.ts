import DefaultTheme from 'vitepress/theme'
import { onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()

    function renderMath() {
      nextTick(() => {
        try {
          const render = (window as any).renderMathInElement
          if (render) {
            render(document.body, {
              delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
              ],
              throwOnError: false,
            })
          }
        } catch (_) {}
      })
    }

    onMounted(renderMath)
    watch(() => route.path, renderMath)
  },
}
