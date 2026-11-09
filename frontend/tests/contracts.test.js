import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Chess } from 'chess.js';
import { getCheckState } from '../src/chessStatus.js';
import formatMarkdown from '../src/formatMarkdown.js';
import { formatApiError } from '../src/apiErrors.js';
import { getNextCourseTopicMeta } from '../src/learningLinks.js';
test('checkmate highlights king of displayed FEN', () => {
 const game = new Chess();
 for (const move of ['f3','e5','g4','Qh4#']) game.move(move);
 assert.deepEqual(getCheckState(game.fen()), {inCheck:true,isCheckmate:true,kingSquare:'e1'});
 assert.equal(getCheckState(new Chess().fen()).inCheck,false);
 assert.equal(getCheckState('invalid').inCheck,false);
});
test('chat escapes HTML and preserves compact markdown', () => {
 assert.equal(formatMarkdown('**Hi**\n\nthere'),'<p><strong>Hi</strong></p><p>there</p>');
 assert.ok(!formatMarkdown('<img src=x onerror=alert(1)>').includes('<img'));
 assert.equal(formatMarkdown('- a\n- b'),'<ul><li>a</li><li>b</li></ul>');
});
test('validation errors remain readable strings', () => {
 assert.equal(typeof formatApiError([{type:'missing',loc:['body','login']}]),'string');
 assert.equal(formatApiError(null,'retry'),'retry');
});
test('next course topic follows the course order and wraps around', () => {
 assert.equal(getNextCourseTopicMeta('pieces').slug, 'king-safety');
 assert.equal(getNextCourseTopicMeta('middlegame').slug, 'pieces');
 assert.equal(getNextCourseTopicMeta('unknown').slug, 'pieces');
 assert.equal(getNextCourseTopicMeta(null).slug, 'pieces');
});
