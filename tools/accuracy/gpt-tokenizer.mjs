// gpt-tokenizer default export uses o200k_base.
import { encode } from 'gpt-tokenizer'

const input = await readAll()
const payload = JSON.parse(input || '{}')
const results = {}

const samples = Array.isArray(payload.samples) ? payload.samples : []
for (const sample of samples) {
  const text = typeof sample.text === 'string' ? sample.text : ''
  const iterations = pickIterations(text.length)
  let count = 0
  encode(text)
  const start = process.hrtime.bigint()
  for (let i = 0; i < iterations; i += 1) {
    count = encode(text).length
  }
  const end = process.hrtime.bigint()
  const avgNs = iterations > 0 ? Number((end - start) / BigInt(iterations)) : 0
  results[sample.name] = { count, avg_ns: avgNs }
}

process.stdout.write(JSON.stringify({ results }))

function pickIterations(size) {
  if (size < 200) return 20000
  if (size < 2000) return 2000
  if (size < 20000) return 200
  if (size < 200000) return 20
  return 5
}

async function readAll() {
  const chunks = []
  for await (const chunk of process.stdin) {
    chunks.push(chunk)
  }
  if (chunks.length === 0) return ''
  return Buffer.concat(chunks).toString('utf8')
}
