const fs = require('fs');
const path = require('path');

const mcpServerSrcDir = 'mcp-server/src';

function fixImportsInFile(filePath) {
    try {
        let content = fs.readFileSync(filePath, 'utf8');
        // Regex para encontrar imports relativos que não terminam em .js
        // Adaptação da regex do Stack Overflow para JavaScript
        const importRegex = /(from\s+['"]\.\.?\/.*?)(['"])/g;
        let newContent = content.replace(importRegex, (match, p1, p2) => {
            // Verifica se o import já termina com .js
            if (p1.endsWith('.js')) {
                return match; // Não modifica se já tem .js
            }
            // Adiciona .js antes da aspas final
            return `${p1}.js${p2}`;
        });

        if (content !== newContent) {
            fs.writeFileSync(filePath, newContent, 'utf8');
            console.log(`Corrigido: ${filePath}`);
        }

    } catch (error) {
        console.error(`Erro ao processar o arquivo ${filePath}: ${error.message}`);
    }
}

function walkDir(dir) {
    const files = fs.readdirSync(dir);

    for (const file of files) {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory()) {
            walkDir(filePath); // Recursivo para subdiretórios
        } else if (stat.isFile() && file.endsWith('.ts')) {
            fixImportsInFile(filePath); // Processa arquivos .ts
        }
    }
}

// Inicia o processo a partir do diretório mcp-server/src
walkDir(mcpServerSrcDir);

console.log('Processo de correção de imports concluído.');