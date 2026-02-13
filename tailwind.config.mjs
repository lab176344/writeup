/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {
			colors: {
				tui: {
					bg: '#1a1b26',       // Deep Blue-Grey
					fg: '#a9b1d6',       // Soft Lavender/Grey (Text)
					accent: '#9ece6a',   // Soft Green (Classic Terminal nod)
					dim: '#565f89',      // Dimmed text
					border: '#414868',   // Border color
					selection: '#33467c', // Selection background
				},
			},
			fontFamily: {
				mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
			},
		},
	},
	plugins: [require('@tailwindcss/typography')],
}
