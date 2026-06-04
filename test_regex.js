const body = `## Contributor Agreement
- [x] I am a human, and not a bot.
- [x] I will be responsible for responding to review comments in a timely manner.
- [x] I will work with the maintainers to push this PR forward until submission.`;

const requiredTerms = [
  'I am a human, and not a bot.',
  'I will be responsible for responding to review comments in a timely manner.',
  'I will work with the maintainers to push this PR forward until submission.'
];

const unchecked = [];
for (const term of requiredTerms) {
  const checkedPattern = new RegExp(`-\\s*\\[\\s*[xX]\\s*\\]\\s*${term.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')}`);
  if (!checkedPattern.test(body)) {
    unchecked.push(term);
  }
}
console.log(unchecked);
