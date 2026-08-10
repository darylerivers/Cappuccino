const fs = require('fs'); const md = MarkdownParser(); try { fs.readFileSync('/app/src/preview.js').toString() } catch (e) { console.error('FAIL') }
