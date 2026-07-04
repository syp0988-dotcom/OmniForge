module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      colors: {
        background: '#111111',
        card: '#1B1B1D',
        hover: '#2A2A2A',
        primary: '#7C5CFC',
        text: '#FFFFFF',
        muted: '#9B9B9B',
        success: '#34C759',
        warning: '#F5A623',
        danger: '#FF3B30'
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular']
      },
      borderRadius: {
        lg: '12px'
      }
    }
  },
  plugins: []
}
