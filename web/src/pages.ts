// The four demo pages, shared by the navigation and the home cards.
export const PAGES = [
  { to: '/will-it-last', icon: '🔋', title: 'Will it last?', short: 'Will it last?', cta: 'Play',
    blurb: "Design a fast-charging recipe and get a lifetime prediction — or an honest refusal." },
  { to: '/promise-vs-reality', icon: '⚖️', title: 'Promise vs reality', short: 'Promise vs reality', cta: 'Compare',
    blurb: "The datasheet's claim drawn over how long the cells actually lasted." },
  { to: '/who-is-lying', icon: '🔍', title: 'Who is lying?', short: 'Who is lying?', cta: 'Investigate',
    blurb: 'Official spec sheets for the same battery that disagree with each other.' },
  { to: '/graph', icon: '🕸️', title: 'The graph', short: 'The graph', cta: 'Explore',
    blurb: 'The similarity network the predictor reasons over, around any tested cell.' },
] as const
