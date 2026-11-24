import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Chess } from 'chess.js';
import { promotionColorForMove } from '../src/promotion.js';

test('promotion is detected for both colours', () => {
  const white = new Chess('4k3/1P6/8/8/8/8/8/4K3 w - - 0 1');
  assert.equal(promotionColorForMove(white, 'b7', 'b8'), 'w');

  const black = new Chess('4k3/8/8/8/8/8/1p6/4K3 b - - 0 1');
  assert.equal(promotionColorForMove(black, 'b2', 'b1'), 'b');
});

test('only pawn moves that reach the last rank are promotions', () => {
  const game = new Chess();
  assert.equal(promotionColorForMove(game, 'e2', 'e4'), null);
  assert.equal(promotionColorForMove(game, 'g1', 'f3'), null);
  assert.equal(promotionColorForMove(game, 'e2', 'e5'), null);
  assert.equal(promotionColorForMove(game, 'e4', 'e5'), null);
  assert.equal(promotionColorForMove(game, 'a1', 'a8'), null);
});

test('capture on the last rank is a promotion too', () => {
  const game = new Chess('2r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1');
  assert.equal(promotionColorForMove(game, 'b7', 'c8'), 'w');

  const move = game.move({ from: 'b7', to: 'c8', promotion: 'n' });
  assert.equal(move.promotion, 'n');
  assert.equal(game.get('c8').type, 'n');
  assert.equal(game.get('c8').color, 'w');
});

test('illegal moves never ask for a promotion piece', () => {
  // Диагональ без фигуры на c8: превращения нет, хотя пешка на 7-й линии.
  const game = new Chess('4k3/1P6/8/8/8/8/8/4K3 w - - 0 1');
  assert.equal(promotionColorForMove(game, 'b7', 'c8'), null);
});
